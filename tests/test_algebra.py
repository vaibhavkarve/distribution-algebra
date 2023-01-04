#!/usr/bin/env python3

from random import random
from typing import TypeAlias

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from distribution_algebra.beta import Beta
from distribution_algebra.config import SAMPLE_SIZE
from distribution_algebra.distribution import (UnivariateDistribution,
                                               VectorizedDistribution)
from distribution_algebra.lognormal import Lognormal
from distribution_algebra.normal import Normal
from distribution_algebra.poisson import Poisson


@given(st.data())
def test_operations(data: st.DataObject) -> None:
    # Normal + Normal ~ Normal
    x: Normal = data.draw(st.from_type(Normal))
    y: Normal = data.draw(st.from_type(Normal))
    assert x + y == Normal(mean=x.mean + y.mean,
                           var=x.var + y.var)

    # Poisson + Poisson ~ Poisson
    z: Poisson = data.draw(st.from_type(Poisson))
    w: Poisson = data.draw(st.from_type(Poisson))
    assert z + w == Poisson(lam=z.lam + w.lam)

    # Lognormal * Lognormal ~ Lognormal
    a: Lognormal = data.draw(st.from_type(Lognormal))
    b: Lognormal = data.draw(st.from_type(Lognormal))
    assert a * b == Lognormal.from_normal_mean_var(a.normal_mean + b.normal_mean,
                                                   a.normal_var + b.normal_var)


UnivariateUnion: TypeAlias = \
    UnivariateDistribution[np.float64] | UnivariateDistribution[np.int_]
VectorizedUnion: TypeAlias = \
    VectorizedDistribution[np.float64] | VectorizedDistribution[np.int_]
DistributionUnion: TypeAlias = UnivariateUnion | VectorizedUnion


@pytest.mark.parametrize("dist1", [Normal(mean=random(), var=random()),
                                   Lognormal(mean=random(), var=random()),
                                   Beta(alpha=random(), beta=random()),
                                   Poisson(lam=random()),
                                   VectorizedDistribution(sample=np.random.random(SAMPLE_SIZE))],
                         ids=["Normal", "Lognormal", "Beta", "Poisson", "Vectorized"])
@pytest.mark.parametrize("dist2", [Normal(mean=random(), var=random()),
                                   Lognormal(mean=random(), var=random()),
                                   Beta(alpha=random(), beta=random()),
                                   Poisson(lam=random()),
                                   VectorizedDistribution(sample=np.random.random(SAMPLE_SIZE))],
                         ids=["Normal", "Lognormal", "Beta", "Poisson", "Vectorized"])
def test_all_pairwise_combinations(dist1: DistributionUnion, dist2: DistributionUnion) -> None:
    addition: UnivariateUnion | VectorizedUnion = dist1 + dist2
    multiplication: UnivariateUnion | VectorizedUnion = dist1 * dist2
    match dist1, dist2:
        case (Normal(), Normal()) | (Poisson(), Poisson()):
            assert isinstance(addition, type(dist1))
            assert isinstance(multiplication, VectorizedDistribution)
        case (Lognormal(), Lognormal()):
            assert isinstance(addition, VectorizedDistribution)
            assert isinstance(multiplication, Lognormal)
        case _:
            assert isinstance(addition, VectorizedDistribution)
            assert isinstance(multiplication, VectorizedDistribution)
