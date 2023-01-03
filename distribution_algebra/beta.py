#!/usr/bin/env python3

from typing import Annotated

import numpy as np
import scipy
from numpy.typing import NDArray
from pydantic import confloat
from pydantic.dataclasses import dataclass

from distribution_algebra.config import RNG
from distribution_algebra.distribution import UnivariateDistribution


@dataclass(frozen=True, kw_only=True, eq=True)
class Beta(UnivariateDistribution[np.float64]):
    alpha: Annotated[float, confloat(gt=0, allow_inf_nan=False)]
    beta: Annotated[float, confloat(gt=0, allow_inf_nan=False)]

    def draw(self, size: int) -> NDArray[np.float64]:
        return RNG.beta(a=self.alpha, b=self.beta, size=size)

    def pdf(self, linspace: NDArray[np.float64]) -> NDArray[np.float64]:
        return scipy.stats.beta.pdf(  # pyright: ignore[reportUnknownVariableType]
            x=linspace, a=self.alpha, b=self.beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def var(self) -> float:
        α: float = self.alpha
        β: float = self.beta
        return α * β / (α + β)**2 / (α + β + 1)

    @property
    def median(self) -> float:
        if self.alpha >= 1 and self.beta >= 1:
            return (self.alpha - 1/3) * (self.alpha + self.beta - 2/3)
        return NotImplemented

    @property
    def mode(self) -> float:
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        if self.alpha <= 1 and self.beta > 1:
            return 0
        if self.alpha > 1 and self.beta <= 1:
            return 1
        return NotImplemented

    @property
    def support(self) -> tuple[float, float]:
        return 0.0, 1.0
