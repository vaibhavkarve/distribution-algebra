#!/usr/bin/env python3

from random import random
from typing import TypeAlias

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from distribution_algebra.beta import Beta
from distribution_algebra.beta4 import Beta4
from distribution_algebra.config import RNG, SAMPLE_SIZE
from distribution_algebra.distribution import (UnivariateDistribution,
                                               VectorizedDistribution)
from distribution_algebra.lognormal import Lognormal
from distribution_algebra.normal import Normal
from distribution_algebra.poisson import Poisson


def randomly_chosen_distributions() -> dict[str,
                                            UnivariateDistribution[np.float64]
                                            | UnivariateDistribution[np.int_]
                                            | VectorizedDistribution[np.float64] | float]:
    return {
        "float": random(),
        "Normal": Normal(mean=random(), var=random()),
        "Lognormal": Lognormal(mean=random(), var=random()),
        "Beta": Beta(alpha=random(), beta=random()),
        "Poisson": Poisson(lam=random()),
        "Beta4": Beta4(alpha=random(), beta=random(), minimum=random(), maximum=random() + 1),
        "VectorizedDistribution (continuous)": VectorizedDistribution(sample=RNG.random(size=SAMPLE_SIZE),
                                                                      is_continuous=True),
    }


def test_distrint_random_choices() -> None:
    for dist1, dist2 in zip(randomly_chosen_distributions().values(),
                            randomly_chosen_distributions().values()):
        assert dist1 != dist2


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

    # 1 - Beta(α, β) ~ Beta(β, α)
    c: Beta = data.draw(st.from_type(Beta))
    assert 1 - c == Beta(alpha=c.beta, beta=c.alpha)
    assert 1 - c == 1.0 - c


UnivariateUnion: TypeAlias = \
    UnivariateDistribution[np.float64] | UnivariateDistribution[np.int_]
VectorizedUnion: TypeAlias = \
    VectorizedDistribution[np.float64] | VectorizedDistribution[np.int_]
DistributionUnion: TypeAlias = UnivariateUnion | VectorizedUnion  # type: ignore[operator]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in power:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in power:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.parametrize("dist1", randomly_chosen_distributions().values(),
                         ids=randomly_chosen_distributions().keys())
@pytest.mark.parametrize("dist2", randomly_chosen_distributions().values(),  # type: ignore
                         ids=randomly_chosen_distributions().keys())
def test_all_pairwise_combinations(dist1: DistributionUnion, dist2: DistributionUnion) -> None:
    addition: UnivariateUnion | VectorizedUnion = dist1 + dist2
    multiplication: UnivariateUnion | VectorizedUnion = dist1 * dist2
    subtraction: UnivariateUnion | VectorizedUnion = dist1 - dist2
    division: UnivariateUnion | VectorizedUnion = dist1 / dist2
    power: UnivariateUnion | VectorizedUnion = dist1 ** dist2
    match dist1, dist2:
        case float(), float():
            assert isinstance(addition, float)
            assert isinstance(multiplication, float)
            assert isinstance(subtraction, float)
            assert isinstance(division, float)
            assert isinstance(power, float)
        case (Normal(), float()) | (float(), Normal()):
            assert isinstance(addition, Normal)
            assert isinstance(multiplication, Normal)
            assert isinstance(subtraction, Normal)
            assert isinstance(division, Normal)
            assert isinstance(power, VectorizedDistribution)
        case Normal(), Normal():
            assert isinstance(addition, Normal)
            assert isinstance(multiplication, VectorizedDistribution)
            assert isinstance(subtraction, Normal)
            assert isinstance(division, VectorizedDistribution)
            assert isinstance(power, VectorizedDistribution)
        case (Lognormal(), float()) | (float(), Lognormal()):
            assert isinstance(addition, VectorizedDistribution)
            assert isinstance(multiplication, Lognormal)
            assert isinstance(subtraction, VectorizedDistribution)
            assert isinstance(division, Lognormal)
            assert isinstance(power, Lognormal)
        case Lognormal(), Lognormal():
            assert isinstance(addition, VectorizedDistribution)
            assert isinstance(multiplication, Lognormal)
            assert isinstance(subtraction, VectorizedDistribution)
            assert isinstance(division, Lognormal)
            assert isinstance(power, VectorizedDistribution)
        case Poisson(), Poisson():
            assert isinstance(addition, Poisson)
            assert isinstance(multiplication, VectorizedDistribution)
            assert isinstance(subtraction, VectorizedDistribution)
            assert isinstance(division, VectorizedDistribution)
            assert isinstance(power, VectorizedDistribution)
        case _:
            assert isinstance(addition, VectorizedDistribution)
            assert isinstance(multiplication, VectorizedDistribution)
            assert isinstance(subtraction, VectorizedDistribution)
            assert isinstance(division, VectorizedDistribution)
            assert isinstance(power, VectorizedDistribution)


@pytest.mark.parametrize("dist", randomly_chosen_distributions().values(),
                         ids=randomly_chosen_distributions().keys())
def test_all_unary_operations(dist: DistributionUnion) -> None:
    negative: UnivariateUnion | VectorizedUnion = - dist
    match dist:
        case float():
            assert isinstance(negative, float)
        case Normal():
            assert isinstance(negative, Normal)
        case Lognormal():
            assert isinstance(negative, VectorizedDistribution)
        case _:
            assert isinstance(negative, VectorizedDistribution)
