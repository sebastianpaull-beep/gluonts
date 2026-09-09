# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .distribution import (
    Distribution,
    _sample_multiple,
    getF,
    nans_like,
)
from .distribution_output import (
    DistributionOutput,
    interval_bounded,
    lower_bounded,
)

# Bounding the degrees of freedom away from the heavy-tail regime keeps the
# mean (needs nu > 1) and the standard deviation (needs nu > 2) defined, and
# keeps sample paths from being dominated by extreme draws.
NU_LOWER_BOUND = 5.0

# sigma is the scale of the *unscaled* target, i.e. before the affine
# transformation applied by `DistributionOutput.distribution`. The lower bound
# avoids a degenerate likelihood, the upper bound keeps the scale on the order
# of the normalised target.
SIGMA_LOWER_BOUND = 1e-3
SIGMA_UPPER_BOUND = 1.0


class StudentT(Distribution):
    r"""
    Student's t-distribution.

    Parameters
    ----------
    mu
        Tensor containing the means, of shape `(*batch_shape, *event_shape)`.
    sigma
        Tensor containing the standard deviations, of shape
        `(*batch_shape, *event_shape)`.
    nu
        Nonnegative tensor containing the degrees of freedom of the
        distribution, of shape `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, mu: Tensor, sigma: Tensor, nu: Tensor, F=None) -> None:
        self.mu = mu
        self.sigma = sigma
        self.nu = nu

    @property
    def F(self):
        return getF(self.mu)

    @property
    def batch_shape(self) -> Tuple:
        return self.mu.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    @property
    def mean(self) -> Tensor:
        return self.F.where(self.nu > 1.0, self.mu, nans_like(self.mu))

    @property
    def stddev(self) -> Tensor:
        F = self.F
        mu, nu, sigma = self.mu, self.nu, self.sigma
        return F.where(nu > 2.0, sigma * F.sqrt(nu / (nu - 2)), nans_like(mu))

    def log_prob(self, x: Tensor) -> Tensor:
        mu, sigma, nu = self.mu, self.sigma, self.nu
        F = self.F

        nup1_half = (nu + 1.0) / 2.0
        part1 = 1.0 / nu * F.square((x - mu) / sigma)
        Z = (
            F.gammaln(nup1_half)
            - F.gammaln(nu / 2.0)
            - 0.5 * F.log(math.pi * nu)
            - F.log(sigma)
        )

        ll = Z - nup1_half * F.log1p(part1)
        return ll

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        def s(mu: Tensor, sigma: Tensor, nu: Tensor) -> Tensor:
            F = self.F
            gammas = F.sample_gamma(
                alpha=nu / 2.0, beta=2.0 / (nu * F.square(sigma)), dtype=dtype
            )
            normal = F.sample_normal(
                mu=mu, sigma=1.0 / F.sqrt(gammas), dtype=dtype
            )
            return normal

        return _sample_multiple(
            s,
            mu=self.mu,
            sigma=self.sigma,
            nu=self.nu,
            num_samples=num_samples,
        )

    @property
    def args(self) -> List:
        return [self.mu, self.sigma, self.nu]


class StudentTOutput(DistributionOutput):
    """
    Student's t output whose scale and degrees of freedom have a bounded
    domain.

    ``mu`` stays unconstrained, ``sigma`` is mapped into
    ``(SIGMA_LOWER_BOUND, SIGMA_UPPER_BOUND)`` by `interval_bounded` and
    ``nu`` into ``(NU_LOWER_BOUND, inf)`` by `lower_bounded`. Both intervals
    are open in exact arithmetic only; see those functions for the behaviour
    at the endpoints in floating point.
    """

    args_dim: Dict[str, int] = {"mu": 1, "sigma": 1, "nu": 1}
    distr_cls: type = StudentT

    @classmethod
    def domain_map(cls, F, mu, sigma, nu):
        sigma = interval_bounded(
            F,
            sigma,
            lower_bound=SIGMA_LOWER_BOUND,
            upper_bound=SIGMA_UPPER_BOUND,
        )
        nu = lower_bounded(F, nu, lower_bound=NU_LOWER_BOUND)
        return mu.squeeze(axis=-1), sigma.squeeze(axis=-1), nu.squeeze(axis=-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
