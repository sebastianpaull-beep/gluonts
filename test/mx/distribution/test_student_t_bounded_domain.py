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

"""
Regression coverage for the bounded parameter domain of the MX
``StudentTOutput``.

``mu`` stays unconstrained, ``sigma`` is squashed into
``(SIGMA_LOWER_BOUND, SIGMA_UPPER_BOUND)`` and ``nu`` into
``(NU_LOWER_BOUND, inf)``. Both intervals are open in exact arithmetic only:
``sigmoid`` and ``softplus`` saturate in floating point, so extreme raw values
do reach the endpoints -- the lower ones exactly, the ``sigma`` upper one only
up to rounding of the rescaling. The bounds are therefore asserted with a
tolerance at the upper end rather than as exact endpoint values.

``sigma`` bounds the *unscaled* parameter: the affine transformation applied by
``DistributionOutput.distribution`` scales the forecast spread on top of it.
"""

import mxnet as mx
import numpy as np
import pytest

from gluonts.mx.distribution import StudentT, StudentTOutput
from gluonts.mx.distribution.distribution_output import (
    interval_bounded,
    lower_bounded,
)
from gluonts.mx.distribution.student_t import (
    NU_LOWER_BOUND,
    SIGMA_LOWER_BOUND,
    SIGMA_UPPER_BOUND,
)

# Raw values where both maps are strictly monotone, sorted ascending.
INTERIOR_RAW_VALUES = [-3.0, -0.5, 0.0, 0.75, 2.5]

# Raw values deep in the saturating regime, kept small enough for ``exp`` not
# to overflow in float32.
EXTREME_RAW_VALUES = [-1e3, -50.0, 50.0]

# Slack for the rounding of ``lower_bound + (upper_bound - lower_bound)``.
BOUND_TOL = 1e-6


def reference_sigma(raw: np.ndarray) -> np.ndarray:
    return SIGMA_LOWER_BOUND + (SIGMA_UPPER_BOUND - SIGMA_LOWER_BOUND) / (
        1.0 + np.exp(-raw)
    )


def reference_nu(raw: np.ndarray) -> np.ndarray:
    return NU_LOWER_BOUND + np.log1p(np.exp(raw))


def column(values, dtype=np.float32) -> mx.nd.NDArray:
    return mx.nd.array(values, dtype=dtype).expand_dims(axis=-1)


def assert_within_bounds(sigma: mx.nd.NDArray, nu: mx.nd.NDArray) -> None:
    sigma, nu = sigma.asnumpy(), nu.asnumpy()

    assert (sigma >= SIGMA_LOWER_BOUND).all()
    assert (sigma <= SIGMA_UPPER_BOUND + BOUND_TOL).all()
    assert (nu >= NU_LOWER_BOUND).all()


def test_bounded_transforms_match_reference_maps():
    raw = mx.nd.array(INTERIOR_RAW_VALUES)
    interior = np.array(INTERIOR_RAW_VALUES)

    bounded_below = lower_bounded(mx.nd, raw, lower_bound=3.0)
    within_interval = interval_bounded(
        mx.nd, raw, lower_bound=-2.0, upper_bound=7.0
    )

    assert np.allclose(
        bounded_below.asnumpy(), 3.0 + np.log1p(np.exp(interior)), rtol=1e-6
    )
    assert np.allclose(
        within_interval.asnumpy(),
        -2.0 + 9.0 / (1.0 + np.exp(-interior)),
        rtol=1e-6,
    )


def test_domain_map_matches_reference_maps_in_interior():
    raw = column(INTERIOR_RAW_VALUES)
    interior = np.array(INTERIOR_RAW_VALUES)

    mu, sigma, nu = StudentTOutput.domain_map(mx.nd, raw, raw, raw)

    assert np.allclose(mu.asnumpy(), interior, rtol=1e-6)
    assert np.allclose(sigma.asnumpy(), reference_sigma(interior), rtol=1e-6)
    assert np.allclose(nu.asnumpy(), reference_nu(interior), rtol=1e-6)

    # away from saturation the maps are strictly inside the open intervals
    assert (sigma.asnumpy() > SIGMA_LOWER_BOUND).all()
    assert (sigma.asnumpy() < SIGMA_UPPER_BOUND).all()
    assert (nu.asnumpy() > NU_LOWER_BOUND).all()


def test_domain_map_stays_within_bounds_for_extreme_raw_values():
    raw = column(EXTREME_RAW_VALUES)

    mu, sigma, nu = StudentTOutput.domain_map(mx.nd, raw, raw, raw)

    assert np.isfinite(sigma.asnumpy()).all()
    assert np.isfinite(nu.asnumpy()).all()
    assert_within_bounds(sigma, nu)
    # mu is deliberately left unconstrained
    assert np.allclose(mu.asnumpy(), np.array(EXTREME_RAW_VALUES))


@pytest.mark.parametrize("raw_value", EXTREME_RAW_VALUES)
def test_loss_gradients_are_finite_for_extreme_raw_values(raw_value):
    raws = [mx.nd.full((4, 1), raw_value) for _ in range(3)]
    for raw in raws:
        raw.attach_grad()

    x = mx.nd.array([-3.0, 0.0, 1.0, 7.0])

    with mx.autograd.record():
        mu, sigma, nu = StudentTOutput.domain_map(mx.nd, *raws)
        loss = StudentT(mu, sigma, nu).loss(x)
        # the reduction has to be recorded too, otherwise the scalar it
        # returns is not part of the graph that `backward` walks
        total_loss = loss.sum()
    total_loss.backward()

    assert np.isfinite(loss.asnumpy()).all()
    for raw in raws:
        assert np.isfinite(raw.grad.asnumpy()).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hybridized_args_proj_preserves_dtype_and_bounds(dtype):
    distr_output = StudentTOutput()
    distr_output.dtype = dtype
    args_proj = distr_output.get_args_proj()
    args_proj.initialize()
    args_proj.hybridize()

    # scaled up to push the projection towards the saturating regime
    mu, sigma, nu = args_proj(
        20.0 * mx.nd.random.normal(shape=(3, 4, 5, 6), dtype=dtype)
    )

    for param in (mu, sigma, nu):
        assert param.dtype == dtype
    assert_within_bounds(sigma, nu)


def test_affine_scaling_applies_on_top_of_bounded_sigma():
    distr_output = StudentTOutput()
    # sigma pinned near its lower bound, so that only the affine scale can
    # produce a forecast spread beyond the sigma upper bound
    args = distr_output.domain_map(
        mx.nd,
        mx.nd.zeros((3, 1)),
        mx.nd.full((3, 1), -8.0),
        mx.nd.zeros((3, 1)),
    )
    loc = 3.0 * mx.nd.ones(shape=(3,))
    scale = 1000.0 * mx.nd.ones(shape=(3,))

    base = distr_output.distribution(args)
    scaled = distr_output.distribution(args, loc=loc, scale=scale)

    assert np.allclose(
        scaled.stddev.asnumpy(), 1000.0 * base.stddev.asnumpy(), rtol=1e-5
    )
    assert np.allclose(
        scaled.mean.asnumpy(),
        1000.0 * base.mean.asnumpy() + 3.0,
        rtol=1e-5,
        atol=1e-6,
    )
    assert (base.stddev.asnumpy() < SIGMA_UPPER_BOUND).all()
    assert (scaled.stddev.asnumpy() > SIGMA_UPPER_BOUND).all()
