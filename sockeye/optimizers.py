# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Extensions to MXNet optimizers
"""

import math

from mxnet.ndarray import NDArray, clip
from mxnet.ndarray import adam_update
from mxnet.optimizer import Optimizer, Adam
from mxnet.random import normal


# convenience wrapper for Optimizer.Register
register = Optimizer.register   # pylint: disable=invalid-name


@register
class AdamPlus(Adam):
    """
    Adam with support for extensions from the following work:

    * Adding Gradient Noise Improves Learning for Very Deep Networks
      Neelakantan et al., (https://arxiv.org/pdf/1511.06807.pdf)

    :param beta1: Exponential decay rate for the first moment estimates.
    :param beta2: Exponential decay rate for the second moment estimates.
    :param epsilon: Small value to avoid division by 0.
    :param noise_eta: Numerator for noise variance.
    :param noise_gamma: Denominator exponent for noise variance.
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 noise_eta: float = 1.,
                 noise_gamma: float = 0.55,
                 **kwargs):
        # Vanilla Adam params
        super().__init__(learning_rate=0.001,
                         beta1=0.9,
                         beta2=0.999,
                         epsilon=1e-8,
                         **kwargs)
        # Extension params
        self.noise_eta = noise_eta
        self.noise_gamma = noise_gamma

    def update(self,
               index: int,
               weight: NDArray,
               grad: NDArray,
               state: object):
        """
        Adapted from MXNet Adam `update()`.

        :param index: The unique index of the parameter into the individual learning
                      rates and weight decays. Learning rates and weight decay
                      may be set via `set_lr_mult()` and `set_wd_mult()`, respectively.
        :param weight: The parameter to be updated.
        :param grad: The gradient of the objective with respect to this parameter.
        :param state: The state returned by `create_state()`.
        """
        assert isinstance(weight, NDArray)
        assert isinstance(grad, NDArray)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        t = self._index_update_count[index]
        coef1 = 1. - self.beta1**t
        coef2 = 1. - self.beta2**t
        lr *= math.sqrt(coef2) / coef1

        # Ext: clip here (outside adam_update) so we can add operations between clipping and update
        if self.clip_gradient:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)   # pylint: disable=E1130

        # Ext: gradient noise
        if self.noise_eta > 0:
            var_t = self.noise_eta / ((1. + t)**self.noise_gamma)
            grad = grad + normal(0., var_t, grad.shape, grad.context)

        mean, var = state
        kwargs = {"beta1": self.beta1,
                  "beta2": self.beta2,
                  "epsilon": self.epsilon,
                  "rescale_grad": self.rescale_grad}
        adam_update(weight, grad, mean, var, out=weight, lr=lr, wd=wd, **kwargs)
