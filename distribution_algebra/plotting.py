#!/usr/bin/env python3

from functools import singledispatch
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from distribution_algebra.beta import Beta
from distribution_algebra.config import SAMPLE_SIZE, SUPPORT_MAX, SUPPORT_MIN
from distribution_algebra.distribution import (T_in, UnivariateDistribution,
                                       VectorizedDistribution)
from distribution_algebra.lognormal import Lognormal
from distribution_algebra.normal import Normal
from distribution_algebra.poisson import Poisson
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from numpy.typing import NDArray


@singledispatch
def plot(*_: Any, ax=None, **kwargs: Any) -> Any:  # pyright: ignore
    raise NotImplementedError

@plot.register(UnivariateDistribution)
def plot_univariate_distribution(
        udist: UnivariateDistribution[T_in], ax=None, **kwargs: Any) -> list[Line2D]:
    ax = ax or plt.gca()
    plot_vectorized_distribution(udist.to_vectorized(), ax=ax, **kwargs)
    if hasattr(udist, "_discrete"):
        arange: NDArray[np.int_] = np.arange(
            start=max(udist.support[0], SUPPORT_MIN),
            stop=min(udist.support[1], SUPPORT_MAX))
        return ax.plot(arange, udist.pdf(arange), color="r")  # type: ignore
    linspace: NDArray[np.float64] = np.linspace(
        start=max(udist.support[0], SUPPORT_MIN),
        stop=min(udist.support[1], SUPPORT_MAX),
        num=SAMPLE_SIZE)
    return ax.plot(linspace, udist.pdf(linspace), color="r")  # type: ignore


@plot.register(VectorizedDistribution)
def plot_vectorized_distribution(
        vdist: VectorizedDistribution[T_in], ax = None, **kwargs: Any) \
        -> tuple[NDArray[np.float64], NDArray[np.float64], list[Polygon]]:
    ax = ax or plt.gca()
    if hasattr(vdist, "_discrete"):
        return ax.hist(vdist.sample, bins=100, alpha=0.25, density=False, **kwargs)  # type: ignore
    return ax.hist(vdist.sample, bins=100, alpha=0.25, density=True, **kwargs)  # type: ignore

@logger.catch
def plot_all_distributions() -> None:
    plt.xkcd()
    fig, axes = plt.subplots(2, 2)

    # Normal distributions.
    ax = axes[0][0]
    a: Normal = Normal(mean=-5.0, var=9.0)
    b: Normal = Normal(mean=1.0, var=4.0)
    plot(a, ax=ax, label=f"{a}")
    plot(b, ax=ax, label=f"{b}")
    ax.legend()

    # Lognormal distributions.
    ax = axes[0][1]
    l1: Lognormal = Lognormal(mean=10, var=10.0)
    plot(l1, ax=ax, label=f"{l1}")
    ax.legend()

    # Beta distributions.
    ax = axes[1][0]
    b1: Beta = Beta(alpha=1, beta=1)
    b2: Beta = Beta(alpha=9, beta=3)
    plot(b1, ax=ax, label=f"{b1}")
    plot(b2, ax=ax, label=f"{b2}")
    ax.legend()

    # Poisson distributions.
    ax = axes[1][1]
    p1: Poisson = Poisson(lam=5)
    plot(p1, ax=ax, label=f"{p1}")
    ax.legend()

    plt.suptitle("Plotting various univariate random distributions.")
    plt.show()


if __name__ == '__main__':
    plot_all_distributions()
