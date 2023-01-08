#!/usr/bin/env python3

from collections import Counter as counter
from functools import singledispatch
from typing import Any, Counter

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from numpy.typing import NDArray

from distribution_algebra.beta import Beta
from distribution_algebra.config import SAMPLE_SIZE, SUPPORT_MAX, SUPPORT_MIN
from distribution_algebra.distribution import (T_in, UnivariateDistribution,
                                               VectorizedDistribution)
from distribution_algebra.lognormal import Lognormal
from distribution_algebra.normal import Normal
from distribution_algebra.poisson import Poisson


@singledispatch
def plot(*_: Any, ax=None, **kwargs: Any) -> Any:  # pyright: ignore
    raise NotImplementedError

@plot.register(UnivariateDistribution)
def plot_univariate_distribution(
        udist: UnivariateDistribution[T_in],
        ax: None | plt.Axes = None,
        **kwargs: Any) -> list[Line2D]:
    ax = ax or plt.gca()
    plot_vectorized_distribution(udist.to_vectorized(), ax=ax, **kwargs)
    if not udist.is_continuous:
        arange: NDArray[np.float64] = np.arange(
            start=max(udist.support[0], SUPPORT_MIN),
            stop=min(udist.support[1], SUPPORT_MAX))
        return ax.plot(arange, udist.pdf(arange), "o", color="r")  # type: ignore
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

    if not vdist.is_continuous:
        sample_counter: counter[T_in] = counter(vdist.sample)
        x: NDArray[np.float64] = np.array(list(sample_counter.keys()), dtype=np.float64)
        heights: NDArray[np.float64] = np.array(
            list(sample_counter.values()), dtype=np.float64) / len(vdist.sample)
        return ax.bar(x, heights, alpha=0.25, align="center", width=0.2, **kwargs) # type: ignore


    number_of_bins: int = min(len(set(vdist.sample)), 100)
    return ax.hist(vdist.sample, bins=number_of_bins, alpha=0.25, density=True,
                   align="left", **kwargs)  # type: ignore


def plot_all_distributions() -> None:
    plt.xkcd()
    axes: tuple[tuple[plt.Axes, ...], ...]
    _, axes = plt.subplots(2, 2)

    # Normal distributions.
    ax: plt.Axes = axes[0][0]
    a: Normal = Normal(mean=-5.0, var=9.0)
    b: Normal = Normal(mean=1.0, var=4.0)
    plot(a, ax=ax, label=f"{a}")
    b_plot = plot(b, ax=ax, label=f"{b}")
    b_plot[0].set_label("Prob. density func.")
    ax.legend()

    # Lognormal distributions.
    ax = axes[0][1]
    l1: Lognormal = Lognormal(mean=10, var=10.0)
    l1_plot = plot(l1, ax=ax, label=f"{l1}")
    l1_plot[0].set_label("Prob. density func.")
    ax.legend()

    # Beta distributions.
    ax = axes[1][0]
    b1: Beta = Beta(alpha=1, beta=1)
    b2: Beta = Beta(alpha=9, beta=3)
    plot(b1, ax=ax, label=f"{b1}")
    b2_plot = plot(b2, ax=ax, label=f"{b2}")
    b2_plot[0].set_label("Prob. density func.")
    ax.legend()

    # Poisson distributions.
    ax = axes[1][1]
    ax.set_xticks(range(20))
    p1: Poisson = Poisson(lam=4)
    p1_plot = plot(p1, ax=ax, label=f"{p1}")
    p1_plot[0].set_label("Prob. mass func.")
    ax.legend()

    plt.suptitle("Plotting various univariate random distributions.")
    plt.show()


if __name__ == '__main__':
    plot_all_distributions()
