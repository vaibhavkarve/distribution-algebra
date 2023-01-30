#!/usr/bin/env python3

from typing import Any, cast

import numpy as np
import scipy
from numpy.typing import NDArray
from attr import frozen, field, validators
from distribution_algebra.config import RNG
from distribution_algebra.distribution import UnivariateDistribution


@frozen(kw_only=True)
class Beta(UnivariateDistribution[np.float64]):
    alpha: float = field(validator=validators.gt(0.0))
    beta: float = field(validator=validators.gt(0.0))

    def draw(self, size: int) -> NDArray[np.float64]:
        return RNG.beta(a=self.alpha, b=self.beta, size=size)

    def pdf(self, linspace: NDArray[np.float64]) -> NDArray[np.float64]:  # type: ignore
        return cast(NDArray[np.float64], scipy.stats.beta.pdf(  # pyright: ignore[reportUnknownVariableType]
            x=linspace, a=self.alpha, b=self.beta))

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

    def __rsub__(self, other: Any) -> Any:
        match other:
            case 1:
                return Beta(alpha=self.beta, beta=self.alpha)
            case _:
                return super().__rsub__(other)  # type: ignore
