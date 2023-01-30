#!/usr/bin/env python3
from dataclasses import field
from math import floor, inf
from typing import Any

import numpy as np
import scipy
from attr import field, frozen, validators
from numpy.typing import NDArray

from distribution_algebra.config import RNG
from distribution_algebra.distribution import UnivariateDistribution


@frozen(kw_only=True)
class Poisson(UnivariateDistribution[np.int64]):
    lam: float = field(validator=validators.gt(0.0))  # rate parameter.

    def draw(self, size: int) -> NDArray[np.int_]:
        return RNG.poisson(lam=self.lam, size=size)

    @property
    def mean(self) -> float:
        return self.lam

    @property
    def var(self) -> float:
        return self.lam

    @property
    def median(self) -> float:
        return floor(self.lam + 1/3 - 1/(50 * self.lam))

    @property
    def mode(self) -> float:
        if not self.lam.is_integer():
            return floor(self.lam)
        return NotImplemented

    def pdf(self, arange: NDArray[np.int_]) -> NDArray[np.float64]:  # type: ignore
        return scipy.stats.poisson.pmf(  # type: ignore
            arange, mu=self.lam)

    def __add__(self, other: Any) -> Any:
        match other:
            case Poisson():
                return Poisson(lam=self.lam + other.lam)
            case _:
                return super().__add__(other)

    @property
    def support(self) -> tuple[float, float]:
        return 0, inf
