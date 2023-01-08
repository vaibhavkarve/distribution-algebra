#!/usr/bin/env python3

import logging

from distribution_algebra.beta import Beta
from distribution_algebra.distribution import (UnivariateDistribution,
                                               VectorizedDistribution)
from distribution_algebra.lognormal import Lognormal
from distribution_algebra.normal import Normal
from distribution_algebra.poisson import Poisson

# Declaring this silences mypy warning on `from distribution_algebra import Beta` etc.
__all__ = ["Normal", "Lognormal", "Beta", "Poisson", "UnivariateDistribution",
           "VectorizedDistribution"]

logging.getLogger('matplotlib').setLevel(logging.ERROR)
