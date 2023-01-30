#!/usr/bin/env python3

import numpy as np
from hypothesis import strategies as st

from distribution_algebra.beta import Beta
from distribution_algebra.distribution import UnivariateDistribution
from distribution_algebra.lognormal import Lognormal
from distribution_algebra.normal import Normal
from distribution_algebra.poisson import Poisson

DISTRIBUTIONS: tuple[type[UnivariateDistribution[np.float64]]
                     | type[UnivariateDistribution[np.int_]], ...] = (
    Normal,
    Lognormal,
    Poisson,
    Beta,
)


# Register a search-strategy for Normal distributions.
st.register_type_strategy(
    Normal, st.builds(Normal,
                      mean=st.floats(width=16, allow_infinity=False, allow_nan=False),
                      var=st.floats(width=16, min_value=0.0, exclude_min=True,
                                    allow_infinity=False, allow_nan=False)))


# Register a search-strategy for Poisson distributions.
st.register_type_strategy(
    Poisson, st.builds(Poisson,
                    lam=st.floats(min_value=0, exclude_min=True,
                                  allow_infinity=False, allow_nan=False)))


# Register a search-strategy for Lognormal distributions.
st.register_type_strategy(
    Lognormal, st.builds(Lognormal,
                         mean=st.floats(width=16, min_value=0.0, exclude_min=True,
                                        allow_infinity=False, allow_nan=False),
                         var=st.floats(width=16, min_value=0.0, exclude_min=True,
                                       allow_infinity=False, allow_nan=False)))


# Register a search-strategy for Beta distributions.
st.register_type_strategy(
    Beta, st.builds(Beta,
                    alpha=st.floats(min_value=0.0, exclude_min=True,
                                    allow_infinity=False, allow_nan=False),
                    beta=st.floats(min_value=0.0, exclude_min=True,
                                   allow_infinity=False, allow_nan=False)))
