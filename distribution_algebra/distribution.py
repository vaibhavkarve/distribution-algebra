#!/usr/bin/env python3

from abc import abstractmethod
from dataclasses import dataclass, field
from math import isclose
from typing import Any, Generic, TypeVar
import warnings

import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from distribution_algebra.algebra import Algebra

from distribution_algebra.config import ABS_TOL, SAMPLE_SIZE

T_in = TypeVar("T_in", np.float64, np.int_)


# Note: Not using `pydantic_dataclass` because it does not support
# NDArray validation.
@dataclass(frozen=True, kw_only=True, eq=False, unsafe_hash=True)
class VectorizedDistribution(Algebra, Generic[T_in]):
    sample: NDArray[T_in]
    is_continuous: bool = field(default=True, repr=False)

    @property
    def mean(self) -> np.float64:
        return np.float64(self.sample.mean())

    @property
    def var(self) -> np.float64:
        return np.float64(self.sample.var())

    def __add__(self, other: Any) -> Any:
        match other:
            case VectorizedDistribution():
                return VectorizedDistribution(sample=self.sample + other.sample)
            case float() | int():
                return VectorizedDistribution(sample=self.sample + other)
            case _:
                return NotImplemented

    def __mul__(self, other: Any) -> Any:
        match other:
            case VectorizedDistribution():
                return VectorizedDistribution(sample=self.sample * other.sample)
            case float() | int():
                return VectorizedDistribution(sample=self.sample * other)
            case _:
                return NotImplemented

    def __sub__(self, other: Any) -> Any:
        match other:
            case VectorizedDistribution():
                return VectorizedDistribution(sample=self.sample - other.sample)
            case float() | int():
                return VectorizedDistribution(sample=self.sample - other)
            case _:
                return NotImplemented

    def __pow__(self, other: Any) -> Any:
        match other:
            case VectorizedDistribution():
                if (self.sample < 0).any():
                    warnings.warn("All entries in base array should be positive.")
                if (other.sample == 0).any() and (self.sample == 0).any():
                    warnings.warn("Attempting to compute 0**0 during (base array)**(other array) computation.")
                return VectorizedDistribution(sample=self.sample ** other.sample)
            case float() | int():
                if (self.sample < 0).any():
                    warnings.warn("All entries in base array should be positive.")
                if not other and (self.sample == 0).any():
                    warnings.warn("Attempting to compute 0**0 during (base array)**other computation.")
                return VectorizedDistribution(sample=self.sample ** other)
            case _:
                return NotImplemented

    def __truediv__(self, other: Any) -> Any:
        match other:
            case VectorizedDistribution():
                if (other.sample == 0).any():
                    warnings.warn("All entries in denominator array should be non-zero.")
                return VectorizedDistribution(sample=self.sample / other.sample)
            case float() | int():
                if other == 0:
                    warnings.warn("Attempting to divide a vectorized distribution by zero.")
                return VectorizedDistribution(sample=self.sample / other)
            case _:
                return NotImplemented

    def __neg__(self) -> Any:
        return VectorizedDistribution(sample=-self.sample)


@pydantic_dataclass(frozen=True, kw_only=True, eq=True)
class UnivariateDistribution(Algebra, Generic[T_in]):
    is_continuous: bool = field(default=True, repr=False)

    @property
    @abstractmethod
    def median(self) -> float: ...

    @property
    @abstractmethod
    def mode(self) -> float: ...

    @property
    @abstractmethod
    def support(self) -> tuple[float, float]: ...

    @abstractmethod
    def draw(self, size: int) -> NDArray[T_in]: ...

    def to_vectorized(self) -> VectorizedDistribution[T_in]:
        return VectorizedDistribution(sample=self.draw(size=SAMPLE_SIZE),
                                      is_continuous=self.is_continuous)

    @abstractmethod
    def pdf(self, linspace: NDArray[np.float64] | NDArray[np.int_]) -> NDArray[np.float64]: ...

    def __add__(self, other: Any) -> Any:
        match other:
            case UnivariateDistribution():
                return self.to_vectorized() + other.to_vectorized()
            case _:
                return self.to_vectorized() + other

    def __mul__(self, other: Any) -> Any:
        match other:
            case UnivariateDistribution():
                return self.to_vectorized() * other.to_vectorized()
            case _:
                return self.to_vectorized() * other

    def __sub__(self, other: Any) -> Any:
        match other:
            case UnivariateDistribution():
                return self.to_vectorized() - other.to_vectorized()
            case _:
                return self.to_vectorized() - other

    def __truediv__(self, other: Any) -> Any:
        match other:
            case UnivariateDistribution():
                return self.to_vectorized() / other.to_vectorized()
            case _:
                return self.to_vectorized() / other

    def __pow__(self, other: Any) -> Any:
        match other:
            case UnivariateDistribution():
                return self.to_vectorized() ** other.to_vectorized()
            case _:
                return self.to_vectorized() ** other

    def __neg__(self) -> Any:
        return -self.to_vectorized()


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({vars(self)})"

    def __eq__(self, other: Any) -> bool:
        match other:
            case UnivariateDistribution():
                if not isinstance(other, type(self)):
                    return False
                if not vars(self).keys() == vars(other).keys():
                    return False
                return all(isclose(value, vars(other)[field], abs_tol=ABS_TOL)
                                   for field, value in vars(self).items())
            case _:
                return NotImplemented
