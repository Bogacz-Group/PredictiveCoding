import typing
import warnings
import torch
import torch.nn as nn

from .pc_layer import PCLayer


def gaussian_energy(inputs):

    t = inputs['mu'] - inputs['x']

    return t * t / inputs['log_sigma'].exp() + inputs['log_sigma']


class VarPCLayer(PCLayer):
    """``VarPCLayer``.

        ``VarPCLayer`` specifies a ``PCLayer`` to be preditive coding energy with variance (log_sigma), which can be specified to be trainable or not.

         This is an adaptation from Luca's refactored pc_layer: https://github.com/YuhangSong/general-energy-nets/blob/pc_update/predictive_coding/pc_layer.py
    """

    def __init__(
        self,
        size,
        init_log_sigma=0.0,
        is_trainable_log_sigma=True,
        **kwargs,
    ):
        """Creates a new instance of ``VarPCLayer``.

        Args:
            size: The size of this layer. This is required as variance is created at the start and maintained afterwards, like in creating a normalization layer you need to specify the size.
            init_log_sigma: The initial log_sigma.
            is_trainable_log_sigma: Whether train log_sigma or not.
            kwargs: The keyword arguments to be passed to underlying ``PCLayer``.
        """

        assert (
            "energy_fn" not in list(kwargs.keys())
        ), "The ``energy_fn`` is specified in VarPCLayer. Thus, cannot be specified in kwargs to underlying ``PCLayer``."

        super().__init__(
            energy_fn=gaussian_energy,
            ** kwargs
        )

        assert isinstance(init_log_sigma, float)
        self.init_log_sigma = init_log_sigma

        assert isinstance(is_trainable_log_sigma, bool)
        self.is_trainable_log_sigma = is_trainable_log_sigma

        log_sigma = torch.full(
            size, self.init_log_sigma
        )
        if self.is_trainable_log_sigma == True:
            self.log_sigma = torch.nn.Parameter(log_sigma)
        else:
            self.log_sigma = log_sigma

    def forward(self, mu: torch.Tensor) -> torch.Tensor:

        return super().forward(
            mu=mu,
            energy_fn_additional_inputs={'log_sigma': self.log_sigma},
        )
