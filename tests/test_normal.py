#!/usr/bin/env python3

from hypothesis import given
from hypothesis import strategies as st

from distribution_algebra.distribution import UnivariateDistribution
from distribution_algebra.normal import Normal


@given(st.from_type(Normal),
       st.floats(width=32, allow_infinity=False, allow_nan=False),
       st.floats(width=64, allow_infinity=False, allow_nan=False))
def test_affine(x: Normal, a: float, b: float) -> None:
    scaled_and_shifted: Normal = a * x + b  # type: ignore[assignment]
    assert scaled_and_shifted == Normal(mean=b + x.mean, var=a**2 * x.var)


@given(st.from_type(Normal))
def test_central_measures(x: Normal) -> None:
    assert x.mean == x.median == x.mode
