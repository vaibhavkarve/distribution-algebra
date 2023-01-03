#!/usr/bin/env python3
from math import inf, sqrt
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
class Normal(UnivariateDistribution[np.float64]):
    mean: Annotated[float, confloat(allow_inf_nan=False)]
    var: Annotated[float, confloat(gt=0.0, allow_inf_nan=False)]

    def draw(self, size: int) -> NDArray[np.float64]:
        return RNG.normal(
            loc=self.mean, scale=sqrt(self.var), size=size)

    def pdf(self, linspace: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], scipy.stats.norm.pdf(  # pyright: ignore[reportUnknownVariableType]
            linspace, loc=self.mean, scale=sqrt(self.var)))

    def __rmul__(self, other: float) -> Self:
        return Normal(mean=self.mean, var=self.var * other**2)

    def __add__(self, other: Any) -> Self:
        match other:
            case int() | float():
                return Normal(mean=self.mean + other, var=self.var)
            case Normal():
                return Normal(mean=self.mean + other.mean, var=self.var + other.var)
            case _:
                return super().__add__(other)

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
