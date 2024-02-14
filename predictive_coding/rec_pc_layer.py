import typing
import warnings
import torch
import torch.nn as nn

from .pc_layer import PCLayer

class RecLayer(nn.Module):
    def __init__(self, d, zero_diagonal=True):
        super(RecLayer, self).__init__()
        self.weight = nn.Parameter(torch.zeros(d, d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.zero_diagonal = zero_diagonal  # Indicator to zero out diagonal entries
        
        # Initialize the weight matrix with zeros on the diagonal if required
        if self.zero_diagonal:
            with torch.no_grad():
                self.weight.fill_diagonal_(0)

        # Register a hook to the weight parameter if zero_diagonal is True
        if self.zero_diagonal:
            self.weight.register_hook(self._zero_diagonal_hook)

    def _zero_diagonal_hook(self, grad):
        # This hook function zeros out the diagonal elements of the gradient
        if self.zero_diagonal:
            with torch.no_grad():
                grad.fill_diagonal_(0)
        return grad

    def forward(self, x):
        # Apply the weight matrix and add the bias term
        return torch.mm(x, self.weight) + self.bias

class RecPCLayer(PCLayer):
    """Recurrent PCLayer
    
    The key difference of a recurrent PC layer to normal PC layer is that 'mu' depends on the layer's value node 'x'
       so we can't have a nn.Linear layer outside of recurrent PC layer
       but rather let recurrent PC layer to define its own 'mu' and 'x'  
    """

    def __init__(
        self,
        size,
        is_zero_diagonal_Wr=False,
        **kwargs,
    ):
        """Creates a new instance of ``RecPCLayer``.

        Args:
            size: The size of this layer. 
            is_zero_diagonal_Wr: Whether to zero out the diagonal entries of the recurrent weight matrix.
            kwargs: The keyword arguments to be passed to underlying ``PCLayer``.
        """

        # assert (
        #     "energy_fn" not in list(kwargs.keys())
        # ), "The ``energy_fn`` is specified in RecPCLayer. Thus, cannot be specified in kwargs to underlying ``PCLayer``."

        super().__init__(
            # energy_fn=lambda inputs: 0.5 * (inputs['x'] - inputs['mu'])**2,
            **kwargs
        )

        assert isinstance(is_zero_diagonal_Wr, bool)
        self.is_zero_diagonal_Wr = is_zero_diagonal_Wr

        # Define recurrent weight and bias
        # We could also initialze it with 0s, following the way for log sigma
        # Zero initialization seem to work well in my previous experiments
        self.Wr = RecLayer(size, zero_diagonal=is_zero_diagonal_Wr)

        # intialize mode to train
        self.mode = 'train'

    def set_mode(self, mode: str):
        """Set the mode of this layer, which can be either 'train' or 'inference'.

        This is special to recurrent PC layer, since in both supervised and memory tasks,
            we want to fix x to input (memory pattern or label) during training, but sample x from mu/initialize it randomly during testing, and let it relax.

        Also note this is different from model.train() and model.eval() in PyTorch,
            because pc_trainer won't allow us to call train_on_batch() during model.eval().
            Therefore, in either mode, we have to maintain model.train().
        """
        assert mode in ['train', 'inference']
        self.mode = mode

    def forward(self, in_value: torch.Tensor) -> torch.Tensor:
        """Given an input, the recurrent PC layer should compute the 'mu' by itself
        This is unlike the original PC layer, which requires the 'mu' to be passed in.
        """

        # sanitize args
        assert isinstance(in_value, torch.Tensor)

        # training attribut belongs to nn.Module
        if self.mode == 'train':
            
            # during training, we don't want to sample, but rather fix x to the value of the data
            # e.g. for supervised learning, x is the target one-hot vector
            self._is_sample_x = False

            # if we have set _x as a parameter in previous inference stage
            # we should detach it and delete it and reassign it to in_value during training
            if isinstance(self._x, nn.Parameter):
                del self._x
            self._x = in_value
            x = self._x

        else:

            # during testing, we want to initialize x by sampling
            # this can be done by setting in_value to a pseudo input (e.g. a zero tensor)
            # and sample x from in_value by specified sample_x_fn

            # detect cases where sample_x is necessary
            if not self._is_sample_x:

                # case: no initialization
                if self._x is None:

                    warnings.warn(
                        (
                            "The <self._x> has not been initialized yet, run with <pc_layer.set_is_sample_x(True)> first. We will do it for you."
                        ),
                        category=RuntimeWarning
                    )
                    self._is_sample_x = True

                # case: device changed
                elif in_value.device != self._x.device:
                    warnings.warn(
                        (
                            "The device of <self._x> is not consistent with that of <mu>, run with <pc_layer.set_is_sample_x(True)> first. We will do it for you."
                        ),
                        category=RuntimeWarning
                    )
                    self._is_sample_x = True

                # case: size changed
                elif in_value.size() != self._x.size():
                    warnings.warn(
                        (
                            "You have changed the shape of this layer, you should do <pc_layer.set_is_sample_x(True) when changing the shape of this layer. We will do it for you.\n"
                            "This should have been taken care of by <pc_trainer> unless you have set <is_sample_x_at_epoch_start=False> when calling <pc_trainer.train_on_batch()>,\n"
                            "in which case you should be responsible for making sure the batch size stays still."
                        ),
                        category=RuntimeWarning
                    )
                    self._is_sample_x = True

            # sample_x
            if self._is_sample_x:

                x_data = self._sample_x_fn(
                    {
                        # 'sample' (in this case, set) x directly from in_value during testing
                        # in memory task, in_value is a corrupted version of the memory pattern
                        # in supervised task, in_value is a random vector that we wish to relax to the target one-hot vector

                        'mu': in_value, 
                        'x': self._x,
                    }
                )

                # make it a parameter
                self._x = nn.Parameter(x_data.to(in_value.device), True)

                # _is_sample_x only takes effect for one pass
                self._is_sample_x = False

            x = self._x
        
        # define mu as self-recurrent, which is a function of x
        mu = self.Wr(x)

        energy_fn_inputs = {
            'x': x,
            'mu': mu,
        }

        energy = self._energy_fn(energy_fn_inputs, **self._energy_fn_kwargs)

        if self.is_keep_energy_per_datapoint:
            # energy, keep the batch dim, other dimensions are reduced to a single dimension
            self._energy_per_datapoint = energy.sum(
                dim=list(
                    range(
                        energy.dim()
                    )
                )[1:],
                keepdim=False,
            ).unsqueeze(1)
            # [batch_size, 1]

        self._energy = energy.sum()

        if self.is_holding_error:
            self.error = (self._x.data - mu).detach().clone()

        return x



