#!/usr/bin/env python3

from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import confloat, validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from distribution_algebra.distribution import UnivariateDistribution
from distribution_algebra.beta import Beta


@pydantic_dataclass(frozen=True, kw_only=True, eq=True)
class Beta4(UnivariateDistribution[np.float64]):
    alpha: Annotated[float, confloat(gt=0, allow_inf_nan=False)]
    beta: Annotated[float, confloat(gt=0, allow_inf_nan=False)]
    minimum: Annotated[float, confloat(allow_inf_nan=False)]
    maximum: Annotated[float, confloat(allow_inf_nan=False)]

    @validator("maximum", pre=True, always=True)
    def maximum_gt_minimum(cls, maximum: float, values: dict[str, float]) -> float:
        assert maximum > values["minimum"]
        return maximum

    def beta_of_alpha_beta(self) -> Beta:
        return Beta(alpha=self.alpha, beta=self.beta)

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
    def support(self) -> tuple[float, float]:
        return self.minimum, self.maximum
