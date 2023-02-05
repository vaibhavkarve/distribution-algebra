#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from attr import Attribute, field, frozen, validators
from numpy.typing import NDArray

from distribution_algebra.beta import Beta
from distribution_algebra.distribution import UnivariateDistribution


@frozen(kw_only=True)
class Beta4(UnivariateDistribution[np.float64]):
    alpha: float = field(validator=validators.gt(0.0))
    beta: float = field(validator=validators.gt(0.0))
    minimum: float = field()
    maximum: float = field()

    @maximum.validator  # type: ignore
    def maximum_gt_minimum(self, _: Attribute["float"], maximum_value: float) -> None:
        assert maximum_value > self.minimum

    def beta_of_alpha_beta(self) -> Beta:
        return Beta(alpha=self.alpha, beta=self.beta)  # pyright: ignore

    def draw(self, size: int) -> NDArray[np.float64]:
        return self.beta_of_alpha_beta().draw(size=size) * (self.maximum - self.minimum) + self.minimum

    def pdf(self, linspace: NDArray[np.float64]) -> NDArray[np.float64]:  # type: ignore
        linspace = (linspace - self.minimum) / (self.maximum - self.minimum)
        return self.beta_of_alpha_beta().pdf(linspace) / (self.maximum - self.minimum)

    @property
    def mean(self) -> float:
        return self.beta_of_alpha_beta().mean * (self.maximum - self.minimum) + self.minimum

    @property
    def var(self) -> float:
        return self.beta_of_alpha_beta().var * (self.maximum - self.minimum)**2

    @property
    def median(self) -> float:
        return self.beta_of_alpha_beta().median * (self.maximum - self.minimum) + self.minimum

    @property
    def mode(self) -> float:
        return self.beta_of_alpha_beta().mode * (self.maximum - self.minimum) + self.minimum

    @property
    def support(self) -> tuple[np.float64, np.float64]:
        return np.float64(self.minimum), np.float64(self.maximum)
