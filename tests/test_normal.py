#!/usr/bin/env python3

from hypothesis import assume, given
from hypothesis import strategies as st

from distribution_algebra.normal import Normal


@given(st.from_type(Normal),
       st.floats(width=16, allow_infinity=False, allow_nan=False),
       st.floats(width=64, allow_infinity=False, allow_nan=False))
def test_affine(x: Normal, a: float, b: float) -> None:
    assume(a)
    assert a * x + b == Normal(mean=a * x.mean + b, var=a**2 * x.var)   # type: ignore


@given(st.from_type(Normal))
def test_central_measures(x: Normal) -> None:
    assert x.mean == x.median == x.mode
