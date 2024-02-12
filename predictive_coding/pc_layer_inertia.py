import typing
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from predictive_coding import pc_layer


class PCLayerInertial(pc_layer.PCLayer):
    """PCLayer.

        PCLayer should be inserted between layers where you want the error to be propagated
            in the predictive coding's (PC's) way, instead of the backpropagation's (BP's) way.
    """

    def __init__(
        self,
        energy_fn: typing.Callable = lambda inputs: 0.5 *
            (inputs['mu'] - inputs['x'])**2 + inputs['x']*inputs['v'].detach() + 0.5*1*inputs['v']**2, # default m = 1
        sample_x_fn: typing.Callable = lambda inputs: inputs['mu'].detach(
                ).clone(),
        sample_v_fn: typing.Callable = lambda inputs: torch.zeros_like(inputs['mu']),
        S: torch.Tensor = None,
        M: torch.Tensor = None,
        is_holding_error: bool = False,
        is_keep_energy_per_datapoint: bool = False,
    ):
        """Creates a new instance of ``PCLayer``.

        Behavior of pc_layer:

            If not pc_layer.training: --> i.e., you just called pc_layer.eval()
                It returns the input.

            If pc_layer.training: --> i.e., you just called pc_layer.train()

                If pc_layer.get_is_sample_x(): --> i.e., you just called pc_layer.set_is_sample_x(True)
                    self._x will be sampled according to sample_x_fn.
                Energy will be computed and held.
                self._x will be returned instead of the input.

        Args:
            energy_fn: The fn that specifies the how to compute the energy of error.
                For example, you can use L2 norm as your energy function by setting:
                    energy_fn = lambda inputs: (inputs['mu'] - inputs['x']).norm(2)
                For example, you can use errors only from the layer closer to the output side:
                    energy_fn = lambda inputs: 0.5 * (inputs['mu'] - inputs['x'].detach())**2
            sample_x_fn: The fn that specifies the how to sample x from mu. Sampling x only happens with you are
                    1, in training mode, and
                    2, you have just called pc_layer.set_is_sample_x(True).
                When both above conditions are satisfied, sample_x_fn will be used to sample x from mu, but just for one time, then self._is_sample_x is set to False again.
                Normally, you should not care about controlling when to sample x from mu at this level, the PCLayer level (meaning you don't have to call pc_layer.set_is_sample_x(True) yourself),
                    because PCTrainer has handled this, see arugments <is_sample_x_at_epoch_start> of PCTrainer.train_on_batch().
                For example:
                    If sample_x_fn = lambda inputs: inputs['mu']
                        it means to sample x as mu.
                    If sample_x_fn = lambda inputs: torch.normal(inputs['mu'])
                        it means to sample x from a normal distribution with mean of mu.

            S: The mask that defines how energy is computed between mu and x interactively.
                Setting to [[1,0,...,0],
                            [0,1,...,0],
                            ...
                            [0,0,...,1]]
                            should make it behave exactly the same as the standard way (setting S to None), i.e. computing
                energy with one-to-one alignment between mu and x.

            M: The mask that select the elements (entries) of energy.
                Setting to [1,1,...,1]
                            should make it behave exactly the same as the standard way (setting M to None), i.e. using all elements (entries) of the energy.

            If both S and M are set to be not None, then S will override the behavior of M.

            is_holding_error: Whether hold the error from mu to x or not.

            is_keep_energy_per_datapoint: if keep energy per datapoint (can get via self.energy_per_datapoint()).
        """

        super().__init__(energy_fn=energy_fn, sample_x_fn=sample_x_fn, S=S, M=M, is_holding_error=is_holding_error, is_keep_energy_per_datapoint=is_keep_energy_per_datapoint)

        self._v = None
        self._sample_v_fn = sample_v_fn

    #  GETTERS & SETTERS  ####################################################################################################


    def get_v(self) -> nn.Parameter:
        """
        """
        return self._v

    #  METHODS  ##############################################################################################################


    def forward(
        self,
        mu: torch.Tensor,
        energy_fn_additional_inputs: dict = {},
    ) -> torch.Tensor:
        """Forward.

        Args:
            mu: The input.

            energy_fn_additional_inputs:
                Additional inputs to be passed to energy_fn.

        Returns:
            The output.
        """

        # sanitize args
        assert isinstance(mu, torch.Tensor)
        assert isinstance(energy_fn_additional_inputs, dict)


        if self.training:

            # detect cases where sample_x is necessary
            if not self._is_sample_x:

                # case: no initialization
                if (self._x is None) or (self._v is None):

                    warnings.warn(
                        (
                            "The <self._x> has not been initialized yet, run with <pc_layer.set_is_sample_x(True)> first. We will do it for you."
                        ),
                        category=RuntimeWarning
                    )
                    self._is_sample_x = True

                # case: device changed
                elif mu.device != self._x.device:
                    warnings.warn(
                        (
                            "The device of <self._x> is not consistent with that of <mu>, run with <pc_layer.set_is_sample_x(True)> first. We will do it for you."
                        ),
                        category=RuntimeWarning
                    )
                    self._is_sample_x = True

                # case: size changed
                elif mu.size() != self._x.size():
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
                        'mu': mu,
                        'x': self._x,
                    }
                )
                v_data = self._sample_v_fn(
                    {
                        'mu': mu,
                        'v': self._v,
                    }
                )

                self._x = nn.Parameter(x_data.to(mu.device), True)
                self._v = nn.Parameter(v_data.to(mu.device), True)

                # _is_sample_x only takes effect for one pass
                self._is_sample_x = False

            x = self._x
            v = self._v

            if self._S is not None:

                # this only works for linear networks
                assert mu.dim() == 2
                assert x.dim() == 2

                size_mu = mu.size(1)
                size_x = x.size(1)

                # self._S.size() = [size_mu, size_x]
                assert self._S.size(0) == size_mu
                assert self._S.size(1) == size_x

                # expand mu
                # [batch_size, size_mu]
                mu = mu.unsqueeze(
                    2
                    # [batch_size, size_mu, 1]
                ).expand(-1, -1, size_x)
                # [batch_size, size_mu, size_x]

                # expand x
                # [batch_size, size_x]
                x = x.unsqueeze(
                    1
                    # [batch_size, 1, size_x]
                ).expand(-1, size_mu, -1)
                # [batch_size, size_mu, size_x]

            energy_fn_inputs = {
                'mu': mu,
                'x': x,
                'v': v,
            }
            energy_fn_inputs.update(energy_fn_additional_inputs)

            energy = self._energy_fn(energy_fn_inputs)

            if self._S is not None:
                # [batch_size, size_mu, size_x]
                energy = energy * self._S.unsqueeze(0)

            elif self._M is not None:

                # [batch_size, size_mu_or_x]
                energy = energy * self._M.unsqueeze(0)

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

            return self._x

        else:

            return mu

    #  PRIVATE METHODS  ######################################################################################################
