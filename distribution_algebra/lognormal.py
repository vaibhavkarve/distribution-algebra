#!/usr/bin/env python3

from math import (exp, expm1, inf, isclose, isfinite, log, log1p, nextafter,
                  sqrt)
from typing import Any, cast

import numpy as np
import scipy
from attr import cmp_using, field, frozen, validators
from numpy.typing import NDArray
from typing_extensions import Self

from distribution_algebra.config import Config
from distribution_algebra.distribution import UnivariateDistribution


@frozen(kw_only=True)
class Lognormal(UnivariateDistribution[np.float64]):
    mean: float = field(
        validator=validators.gt(0.0),
        eq=cmp_using(eq=lambda a, b: isclose(a, b, abs_tol=Config.abs_tol)),  # pyright: ignore
    )
    var: float = field(
        validator=validators.gt(0.0),
        eq=cmp_using(eq=lambda a, b: isclose(a, b, abs_tol=Config.abs_tol)),  # pyright: ignore
    )

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
    def support(self) -> tuple[np.float64, np.float64]:
        return np.float64(nextafter(0.0, inf)), np.float64(np.finfo(np.float64).max)

    @classmethod
    def from_normal_mean_var(cls, μ: float, σ̂2: float) -> Self:
        var = expm1(σ̂2) * exp(2 * μ + σ̂2)
        assert isfinite(var)
        return cls(
            mean=exp(μ + σ̂2 / 2), var=expm1(σ̂2) * exp(2 * μ + σ̂2)
        )  # pyright: ignore

    def draw(self, size: int) -> NDArray[np.float64]:
        return Config.rng.lognormal(
            mean=self.normal_mean, sigma=sqrt(self.normal_var), size=size
        )

    def pdf(self, linspace: NDArray[np.float64]) -> NDArray[np.float64]:  # type: ignore
        return cast(
            NDArray[np.float64],
            scipy.stats.lognorm.pdf(  # pyright: ignore[reportUnknownVariableType]
                x=linspace,
                s=sqrt(self.normal_var),
                loc=0.0,
                scale=self.mean**2 / sqrt(self.mean**2 + self.var),
            ),
        )

    def __mul__(self, other: Any) -> Any:
        match other:
            case int() | float() if other > 0:
                return Lognormal.from_normal_mean_var(
                    self.normal_mean + log(other), self.normal_var
                )
            case int() | float():
                raise ValueError(
                    f"Can only multiply Lognormal distribution {self}"
                    f" with a positive number, not {other}."
                )
            case Lognormal():
                return Lognormal.from_normal_mean_var(
                    self.normal_mean + other.normal_mean,
                    self.normal_var + other.normal_var,
                )
            case _:
                return super().__mul__(other)

    def __truediv__(self, other: Any) -> Any:
        match other:
            case int() | float():
                return self * (1 / other)
            case Lognormal():
                return Lognormal.from_normal_mean_var(
                    self.normal_mean - other.normal_mean,
                    self.normal_var + other.normal_var,
                )
            case _:
                return super().__truediv__(other)

    def __pow__(self, other: Any) -> Any:
        match other:
            case int() | float():
                return Lognormal.from_normal_mean_var(
                    other * self.normal_mean, other**2 * self.normal_var
                )
            case _:
                return super().__pow__(other)
