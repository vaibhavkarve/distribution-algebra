#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isclose
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass

from distribution_algebra.config import ABS_TOL, SAMPLE_SIZE

T_in = TypeVar("T_in", np.float64, np.int_)


@dataclass
class VectorizedDistribution(Generic[T_in]):
    sample: NDArray[T_in]

    @property
    def mean(self) -> np.float64:
        return self.sample.mean()

    @property
    def var(self) -> np.float64:
        return self.sample.var()

    def __add__(self, other: Any) -> Any:
        match other:
            case VectorizedDistribution():
                return VectorizedDistribution(sample=self.sample + other.sample)
            case UnivariateDistribution():
                return VectorizedDistribution(sample=self.sample + other.to_vectorized().sample)
            case _:
                return NotImplemented

    def __mul__(self, other: Any) -> Any:
        match other:
            case VectorizedDistribution():
                return VectorizedDistribution(sample=self.sample * other.sample)
            case UnivariateDistribution():
                return VectorizedDistribution(sample=self.sample * other.to_vectorized().sample)
            case _:
                return NotImplemented


@pydantic_dataclass(frozen=True, kw_only=True, eq=True)
class UnivariateDistribution(ABC, Generic[T_in]):
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
        return VectorizedDistribution(sample=self.draw(size=SAMPLE_SIZE))

    @abstractmethod
    def pdf(self, linspace: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def __add__(self, other: Any) -> Any:
        match other:
            case UnivariateDistribution():
                return VectorizedDistribution(sample=self.to_vectorized().sample
                                              + other.to_vectorized().sample)
            case VectorizedDistribution():
                return VectorizedDistribution(sample=self.to_vectorized().sample
                                              + other.sample)
            case _:
                return NotImplemented

    def __mul__(self, other: Any) -> Any:
        match other:
            case UnivariateDistribution():
                return VectorizedDistribution(sample=self.to_vectorized().sample
                                              * other.to_vectorized().sample)
            case VectorizedDistribution():
                return VectorizedDistribution(sample=self.to_vectorized().sample
                                              * other.sample)
            case _:
                return NotImplemented

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
