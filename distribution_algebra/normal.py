#!/usr/bin/env python3
from math import inf, sqrt
from typing import Annotated, Any, cast

import numpy as np
import scipy
from attr import frozen, field, validators
from numpy.typing import NDArray
from typing_extensions import Self

from distribution_algebra.config import RNG
from distribution_algebra.distribution import UnivariateDistribution


@frozen(kw_only=True)
class Normal(UnivariateDistribution[np.float64]):
    mean: float
    var: float = field(validator=validators.gt(0.0))

    def draw(self, size: int) -> NDArray[np.float64]:
        return RNG.normal(loc=self.mean, scale=sqrt(self.var), size=size)

    def pdf(self, linspace: NDArray[np.float64]) -> NDArray[np.float64]:  # type: ignore
        return cast(NDArray[np.float64], scipy.stats.norm.pdf(  # pyright: ignore[reportUnknownVariableType]
            linspace, loc=self.mean, scale=sqrt(self.var)))

    def __add__(self, other: Any) -> Any:
        match other:
            case Normal():
                return Normal(mean=self.mean + other.mean, var=self.var + other.var)
            case int() | float():
                return Normal(mean=self.mean + other, var=self.var)
            case _:
                return super().__add__(other)

    def __mul__(self, other: Any) -> Any:
        match other:
            case int() | float():
                return Normal(mean=self.mean * other, var=self.var * other**2)
            case _:
                return super().__mul__(other)

    def __sub__(self, other: Any) -> Any:
        match other:
            case Normal():
                return Normal(mean=self.mean - other.mean, var=self.var + other.var)
            case int() | float():
                return Normal(mean=self.mean - other, var=self.var)
            case _:
                return super().__sub__(other)

    def __truediv__(self, other: Any) -> Any:
        match other:
            case int() | float():
                return Normal(mean=self.mean / other, var=self.var / other**2)
            case _:
                return super().__truediv__(other)

    def __neg__(self) -> Self:
        return Normal(mean=-self.mean, var=self.var)

    @property
    def median(self) -> float:
        return self.mean

    @property
    def mode(self) -> float:
        return self.mean

    @property
    def support(self) -> tuple[float, float]:
        return -inf, inf

    @classmethod
    def from_sample(cls, sample: NDArray[np.float64]) -> Self:
        sample_mean: float = sample.mean()
        sample_variance: float = sample.var()
        sample_length: int = len(sample)
        unbiased_variance = sample_length * sample_variance / (sample_length - 1)
        return cls(mean=sample_mean, var=unbiased_variance)
