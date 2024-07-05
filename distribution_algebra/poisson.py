#!/usr/bin/env python3
from math import floor
from typing import Any

import attr
import numpy as np
import scipy
from numpy.typing import NDArray

from distribution_algebra.config import Config
from distribution_algebra.distribution import UnivariateDistribution


@attr.frozen(kw_only=True)
class Poisson(UnivariateDistribution[np.int_]):
    lam: float = attr.field(validator=attr.validators.gt(0.0))  # rate parameter.

    def draw(self, size: int) -> NDArray[np.int_]:
        return Config.rng.poisson(lam=self.lam, size=size)

    @property
    def mean(self) -> float:
        return self.lam

    @property
    def var(self) -> float:
        return self.lam

    @property
    def median(self) -> float:
        return floor(self.lam + 1 / 3 - 1 / (50 * self.lam))

    @property
    def mode(self) -> float:
        if not self.lam.is_integer():
            return floor(self.lam)
        return NotImplemented

    def pdf(self, arange: NDArray[np.int_]) -> NDArray[np.float64]:  # type: ignore
        return scipy.stats.poisson.pmf(arange, mu=self.lam)  # type: ignore

    def __add__(self, other: Any) -> Any:
        match other:
            case Poisson():
                return Poisson(lam=self.lam + other.lam)  # pyright: ignore
            case _:
                return super().__add__(other)

    @property
    def support(self) -> tuple[np.int_, np.int_]:
        return np.int_(0), np.int_(np.iinfo(np.int_).max)
