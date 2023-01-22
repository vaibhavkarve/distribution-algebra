#!/usr/bin/env python3
"""This module defines abstract classes for statistical distributions.

We define 2 types of distributions. The first is
`UnivariateDistribution`. Most named statistical distributions we
learn about in a statistics course (Normal, Beta, Uniform, etc.) can
be thought of as instances of `UnivariateDistribution`.

Second, we define a statistical distribution whose parameter might not
be known. Such a distribution is tracked by keeping around a really
large sample. We name this a `VectorizedDistribution`. In many cases,
working with vectorized distributions is easier since it allows us to
apply algebraic operations to them.

"""

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
from typing import Any


T_in = TypeVar("T_in", np.float64, np.int_)


@dataclass(frozen=True, kw_only=True, eq=False, unsafe_hash=True)
class VectorizedDistribution(Algebra, Generic[T_in]):
    """A Vectorized form of a probability distribution.

    Params:
       sample: the sample-vector used in lieu of a closed-form expression for
          the distribution.
       is_continuous: used to track if the distribution is continuous or discrete.

    Other Params: Implementation details
       * We do not use `pydantic_dataclass` in this definition because it
         does not support NDArray validation.
       * In most cases, if using  `(x: UnivariateDistribution).to_vectorized()`
         to create a `VectorizedDistribution` instance, the instance's `is_continuous`
         parameter is equal to `x.is_continuous`. In other words, continuous
         (discrete) univariate distributions give rise to continuous (discrete)
         vectorized distributions.
       * We set `frozen=True` to enforce immutability of each instance of this class.
         To modify the class or its sample-attribute, the user must define a new
         instance instead of modifying an existing one.
       * We set `unsafe_hash=True` to make this class hashable (it is immutable
         after all).
       * `sample` is chosen to be a numpy array to leverage the fast, vectorized
         operations enabled by numpy's C API.

    """
    sample: NDArray[T_in]
    is_continuous: bool = field(default=True, repr=False)

    @property
    def mean(self) -> np.float64:
        """Mean of the distribution, defined as the mean of the sample.

        Returns:
           mean of the distribution, defined as the mean of the sample.
        """
        return np.float64(self.sample.mean())

    @property
    def var(self) -> np.float64:
        """Variance of the distribution, defined as the variance of the sample.

        Returns:
           variance of the distribution, defined as the variance of the sample.
        """
        return np.float64(self.sample.var())


    def __add__(self, other: Any) -> Any:
        """Return the left-sum of a VectorDistribution with any other type.

        Types:
           - `VectorizedDistribution + VectorizedDistribution`: the result is also a
             vectorized-distrubution whose samples parameter is the element-wise sum
             of the sample parameters of the summands.
           - `VectorizedDistribution + Number`: the result is a vectorized-distribution
             whose sample values have been shifted.
        """
        match other:
            case VectorizedDistribution():
                return VectorizedDistribution(sample=self.sample + other.sample)
            case float() | int():
                return VectorizedDistribution(sample=self.sample + other)
            case _:
                return NotImplemented

    def __mul__(self, other: Any) -> Any:
        """Return the left-product of a VectorDistribution with any other type.

        Types:
           - `VectorizedDistribution * VectorizedDistribution`: the result is also a
             vectorized-distrubution whose samples parameter is the element-wise product
             of the sample parameters of the multiplicands.
           - `VectorizedDistribution * Number`: the result is a vectorized-distribution
             whose sample values have been scaled.
        """
        match other:
            case VectorizedDistribution():
                return VectorizedDistribution(sample=self.sample * other.sample)
            case float() | int():
                return VectorizedDistribution(sample=self.sample * other)
            case _:
                return NotImplemented

    def __sub__(self, other: Any) -> Any:
        """Return the left-difference of a VectorDistribution with any other type.

        Types:
           - `VectorizedDistribution - VectorizedDistribution`: the result is also a
             vectorized-distrubution whose samples parameter is the element-wise difference
             of the sample parameters of the arguments.
           - `VectorizedDistribution - Number`: the result is a vectorized-distribution
             whose sample values have been shifted.
        """
        match other:
            case VectorizedDistribution():
                return VectorizedDistribution(sample=self.sample - other.sample)
            case float() | int():
                return VectorizedDistribution(sample=self.sample - other)
            case _:
                return NotImplemented

    def __pow__(self, other: Any) -> Any:
        """Return the left-power of a VectorDistribution with any other type.

        Types:
           - `VectorizedDistribution ** VectorizedDistribution`: the result is also a
             vectorized-distrubution. Result's i'th sample entry is
             `self.sample[i] ** other.sample[i]`.
           - `VectorizedDistribution - Number`: the result is a vectorized-distribution.
             Result's i'th sample entry is `self.sample[i] ** other`.

        Warns:
           - UserWarning: if computing `VectorizedDistribution ** (VectorizedDistrion | Number)`
             would result in raising a negative number to any power.
           - UserWarning: if computing `VectorizedDistribution ** (VectorizedDistribution | Number)`
             would result in a `0 ** 0` computation.
        """
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
        """Return the left-division of a VectorDistribution with any other type.

        Types:
           - `VectorizedDistribution / VectorizedDistribution`: the result is also a
             vectorized-distrubution whose sample parameter is the element-wise ratio
             of the sample parameters of the arguments.
           - `VectorizedDistribution / Number`: the result is a vectorized-distribution
             whose sample parameter has been scaled down.

        Warns:
           - UserWarning: if computing `VectorizedDistribution / (VectorizedDistrion | Number)`
             would result in a `x / 0` calculation, for any x.
        """
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
        """Return the negative of a VectorizedDistribution.

        Returns:
           (VectorizedDistribution): A VectorizedDistribution whose sample entries are the
              negative of the original's.
        """
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
