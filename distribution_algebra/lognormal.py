#!/usr/bin/env python3

from math import exp, expm1, inf, isfinite, log, log1p, nextafter, sqrt
from typing import Annotated, Any, cast

import numpy as np
import scipy
from numpy.typing import NDArray
from pydantic import confloat
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from distribution_algebra.config import RNG, SAMPLE_SIZE
from distribution_algebra.distribution import (UnivariateDistribution,
                                               VectorizedDistribution)


@dataclass(frozen=True, kw_only=True, eq=False)
class Lognormal(UnivariateDistribution[np.float64]):
    mean: Annotated[float, confloat(gt=0.0, allow_inf_nan=False)]
    var: Annotated[float, confloat(gt=0.0, allow_inf_nan=False)]

    @property
    def normal_mean(self) -> float:
        # log(μ² / sqrt(μ² + σ²))
        return 2 * log(self.mean) - log(self.mean**2 + self.var) / 2

    @property
    def normal_var(self) -> float:
        return log1p(self.var / self.mean**2)

    @property
    def median(self) -> float:
        return exp(self.normal_mean)

    @property
    def mode(self) -> float:
        return exp(self.normal_mean - self.normal_var)

    @property
    def support(self) -> tuple[float, float]:
        return nextafter(0.0, inf), inf

    @classmethod
    def from_normal_mean_var(cls, μ: float, σ̂2: float) -> Self:
        var = expm1(σ̂2) * exp(2*μ + σ̂2)
        assert isfinite(var)
        return_value = cls(mean=exp(μ + σ̂2 / 2),
                           var=expm1(σ̂2) * exp(2*μ + σ̂2))
        return_value.__pydantic_validate_values__()
        return return_value

    def draw(self, size: int) -> NDArray[np.float64]:
        return RNG.lognormal(mean=self.normal_mean, sigma=sqrt(self.normal_var), size=size)

    def pdf(self, linspace: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast(NDArray[np.float64],
                    scipy.stats.lognorm.pdf(  # pyright: ignore[reportUnknownVariableType]
                        x=linspace, s=sqrt(self.normal_var),
                        loc=0.0,
                        scale=self.mean**2 / sqrt(self.mean**2 + self.var),
            ))

    def __pow__(self, other: Any) -> Self:
        match other:
            case int() | float():
                return Lognormal.from_normal_mean_var(other * self.normal_mean,
                                                      other**2 * self.normal_var)
            case _:
                return NotImplemented

    def __rmul__(self, other: Any) -> Self:
        match other:
            case int() | float() if other > 0:
                return Lognormal.from_normal_mean_var(self.normal_mean + log(other),
                                                      self.normal_var)
            case int() | float():
                raise ValueError(f"Can only multiply Lognormal distribution {self}"
                                 f" with a positive number, not {other}.")
            case _:
                return NotImplemented

    def __mul__(self, other: Any) -> Any:
            match other:
                case Lognormal():
                    return Lognormal.from_normal_mean_var(self.normal_mean + other.normal_mean,
                                                          self.normal_var + other.normal_var)
                case _:
                    return super().__mul__(other)
