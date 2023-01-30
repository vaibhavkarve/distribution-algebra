#!/usr/bin/env python3

from math import isclose

from hypothesis import assume, given
from hypothesis import strategies as st
from numpy import log

from distribution_algebra.config import ABS_TOL
from distribution_algebra.lognormal import Lognormal


@given(st.from_type(Lognormal))
def test_lognormal_to_normal_to_lognormal(x: Lognormal) -> None:
    assert x == Lognormal.from_normal_mean_var(x.normal_mean, x.normal_var)


@given(st.from_type(Lognormal),
       st.floats(min_value=-2, max_value=2, width=16, allow_infinity=False, allow_nan=False),
       st.floats(width=16, allow_infinity=False, allow_nan=False, min_value=0.0, exclude_min=True))
def test_power_and_scaling(x: Lognormal, a: float, b: float) -> None:
    assume(a)
    scaled_and_powered: Lognormal = b * (x ** a)
    assert isclose(scaled_and_powered.normal_mean, a * x.normal_mean + log(b), abs_tol=ABS_TOL)
    assert isclose(scaled_and_powered.normal_var, a**2 * x.normal_var, abs_tol=ABS_TOL), scaled_and_powered.var / scaled_and_powered.mean**2
