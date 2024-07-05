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

import warnings
from abc import abstractmethod
from math import isclose
from typing import Any, Generic, Literal, TypeVar, get_args

import attr
import numpy as np
from numpy.typing import NDArray

from distribution_algebra.algebra import Algebra
from distribution_algebra.config import Config

T_in = TypeVar("T_in", np.float64, np.int_)


@attr.frozen(kw_only=True)
class VectorizedDistribution(Algebra, Generic[T_in]):
    """A Vectorized form of a probability distribution.

    Params:
       sample: the sample-vector used in lieu of a closed-form expression for
          the distribution.
       is_continuous: used to track if the distribution is continuous or discrete.

    Other Params: Implementation details
       * We use `attr.frozen` in this definition because it supports validation and
         immutability.
       * We use `attr.frozen` to enforce immutability of each instance of this class.
         To modify the class or its sample-attribute, the user must define a new
         instance (or use `attr.evolve`) instead of modifying an existing one.
       * `sample` is chosen to be a numpy array to leverage the fast, vectorized
         operations enabled by numpy's C API.

    """

    sample: NDArray[T_in] = attr.field(eq=attr.cmp_using(eq=np.array_equal))  # type: ignore
    is_continuous: bool = attr.field(repr=False)

    @is_continuous.validator  # type: ignore
    def check_is_continuous(
        self, _: Literal["is_continuous"], is_continuous_value: bool
    ) -> None:
        match self.sample.dtype:
            case np.int_:
                assert not is_continuous_value
            case np.float64:
                assert is_continuous_value
            case _:
                raise TypeError(
                    f"Encountered unknown type {self.sample.dtype} in VectorizedDistribution."
                )

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
                return VectorizedDistribution(  # type: ignore
                    sample=self.sample + other.sample,
                    is_continuous=self.is_continuous or other.is_continuous,
                )
            case float() | int():
                return VectorizedDistribution(
                    sample=self.sample + other,  # type: ignore
                    is_continuous=self.is_continuous or isinstance(other, float),
                )
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
                return VectorizedDistribution(  # type: ignore
                    sample=self.sample * other.sample,
                    is_continuous=self.is_continuous or other.is_continuous,
                )
            case float() | int():
                return VectorizedDistribution(  # type: ignore
                    sample=self.sample * other,
                    is_continuous=self.is_continuous or isinstance(other, float),
                )
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
                return VectorizedDistribution(  # type: ignore
                    sample=self.sample - other.sample,
                    is_continuous=self.is_continuous or other.is_continuous,
                )
            case float() | int():
                return VectorizedDistribution(  # type: ignore
                    sample=self.sample - other,
                    is_continuous=self.is_continuous or isinstance(other, float),
                )
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
                    warnings.warn(
                        "Attempting to compute 0**0 during (base array)**(other array) computation."
                    )
                return VectorizedDistribution(  # type: ignore
                    sample=self.sample**other.sample,
                    is_continuous=self.is_continuous or other.is_continuous,
                )
            case float() | int():
                if (self.sample < 0).any():
                    warnings.warn("All entries in base array should be positive.")
                if not other and (self.sample == 0).any():
                    warnings.warn(
                        "Attempting to compute 0**0 during (base array)**other computation."
                    )
                return VectorizedDistribution(  # type: ignore
                    sample=self.sample**other,
                    is_continuous=self.is_continuous or isinstance(other, float),
                )
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
                    warnings.warn(
                        "All entries in denominator array should be non-zero."
                    )
                return VectorizedDistribution(  # type: ignore
                    sample=self.sample / other.sample, is_continuous=True
                )
            case float() | int():
                if other == 0:
                    warnings.warn(
                        "Attempting to divide a vectorized distribution by zero."
                    )
                return VectorizedDistribution(  # type: ignore
                    sample=self.sample / other, is_continuous=True
                )
            case _:
                return NotImplemented

    def __neg__(self) -> Any:
        """Return the negative of a VectorizedDistribution.

        Returns:
           (VectorizedDistribution): A VectorizedDistribution whose sample entries are the
              negative of the original's.
        """
        return VectorizedDistribution(sample=-self.sample,
                                      is_continuous=self.is_continuous)  # type: ignore


@attr.frozen(kw_only=True)
class UnivariateDistribution(Algebra, Generic[T_in]):
    """A univariate probability distribution.

    All named distributions defined in this library are subclasses of
    `UnivariateDistribution`. This class can also be used by a user for defining new
    distributions (as subclasses) but beware that the class mandates
    lots of properties and methods be defined for each concrete subclass.

    Params:
       is_continuous: used to track if the distribution is continuous or discrete.

    Other Params: Implementation details
       * If `is_continuous` is set to True, then `T_in` should be a "continuous" type like
         `float` or `numpy.float64`. If it is set to False, then `T_in` should be a discrete
         type like `int` or `numpy.int_`.
       * We set `frozen=True` to enforce immutability of each subclass and each instance of
         this class.
       * We set `unsafe_hash=True` to make this class hashable (it is immutable
         after all).
    """

    @property
    def is_continuous(self) -> bool:
        """Return True if the distribution is continuous, else return False.

        Continuity (or distcreteness) of the distribution is the
        continuity (or discreteness) of its support space. This is
        captured by the Generic type variable `T_in`. If the
        distribution is continuous (like `Normal <:
        UnivariateDistribution[numpy.float64]`, the this property returns
        True. If the distribution is discrete (like `Binomial <:
        UnivariateDistribution[numpy.int_]`, then this property is False.

        """
        assert hasattr(self, "__orig_bases__")
        T_in_at_runtime: type = get_args(self.__orig_bases__[0])[0]  # pyright: ignore
        match T_in_at_runtime:
            case np.int_ | np.int_:
                return False
            case np.float64:
                return True
            case _:
                raise TypeError(f"Found unsupported type {T_in_at_runtime}")

    @property
    @abstractmethod
    def median(self) -> float:
        """Return the median value of the distribution if it is defined."""

    @property
    @abstractmethod
    def mode(self) -> float:
        """Return the mode (most-liekely) value of the distribution if it is defined."""

    @property
    @abstractmethod
    def support(self) -> tuple[T_in, T_in]:
        """Return the (min, max) range of the distribution's support interval.

        In some cases, if one of the support ends is positive or
        negative infinity, we use the closest value to infinity
        expressible while using a constrained `T_in` type.

        """

    @abstractmethod
    def draw(self, size: int) -> NDArray[T_in]:
        """Draw a random sample of given `size` from the distribution.

        For efficiently drawing many random numbers from a
        distribution with fixed parameters, it is recommended that
        this method be used with the appropriate `size` instead of
        drawing `size=1` samples inside a separate loop.

        Note:
           Not to be confused with `distribution_algebra.plotter.plot`
           method which is used for plotting a distribution's graph.

        Implementation detail:
           This method should always use numpy's `default_rng` and its
           associated methods for drawing random samples from various
           distributions.

        """

    def to_vectorized(self) -> VectorizedDistribution[T_in]:
        """Convert to a `VectorizedDistribution`.

        This method is called internally when we perform algebraic
        operations on distributions whose closed-form expressions are
        not known.

        Note:
           In most cases, if using `(x:
           UnivariateDistribution).to_vectorized()` to create a
           `VectorizedDistribution` instance, the instance's
           `is_continuous` parameter is equal to `x.is_continuous`. In
           other words, continuous (discrete) univariate distributions
           give rise to continuous (discrete) vectorized
           distributions.

        """
        return VectorizedDistribution(  # type: ignore
            sample=self.draw(size=Config.sample_size), is_continuous=self.is_continuous
        )

    @abstractmethod
    def pdf(self, linspace: NDArray[T_in]) -> NDArray[np.float64]:
        """Return the probability density (or mass) function's values at the given domain points.

        Implementation detail:
           This method usually uses scipy's statistical distribution's
           probability density/mass functions.

        In the case of continuous (or discrete) distributions, this
        function accepts a parameter vector of points in
        `numpy.float64` (or `numpy.int_`) space. While the function's
        implementation does not put any constraints on the input
        vector, the user can usually generate an appropriate input by
        using `numpy.linspace` or `numpy.arange`.

        """

    def __add__(self, other: Any) -> Any:
        """Return the left-sum of a UnivatiateDistribution with any other type.

        Types:
           - `UnivariateDistribution + UnivariateDistribution`: the result is the sum of
             the vectorized versions of the two distributions.
           - `UnivariateDistribution + Any`: the left-parameter is vectorized and then
             added to the right-parameter.

        """
        match other:
            case UnivariateDistribution():
                return self.to_vectorized() + other.to_vectorized()
            case _:
                return self.to_vectorized() + other

    def __mul__(self, other: Any) -> Any:
        """Return the left-product of a UnivariateDistribution with any other type.

        Types:
           - `UnivariateDistribution * UnivariateDistribution`: the result is the product of
             the vectorized versions of the two distributions.
           - `UnivariateDistribution * Any`: the left-parameter is vectorized and then
             multiplied by the right-parameter.
        """
        match other:
            case UnivariateDistribution():
                return self.to_vectorized() * other.to_vectorized()
            case _:
                return self.to_vectorized() * other

    def __sub__(self, other: Any) -> Any:
        """Return the left-difference of a UnivariateDistribution with any other type.

        Types:
           - `UnivariateDistribution - UnivariateDistribution`: the result is the
             difference of the vectorized versions of the two distributions.
           - `UnivariateDistribution - Any`: the left-parameter is vectorized and then
             the right-parameter is subtracted from it.
        """
        match other:
            case UnivariateDistribution():
                return self.to_vectorized() - other.to_vectorized()
            case _:
                return self.to_vectorized() - other

    def __truediv__(self, other: Any) -> Any:
        """Return the left-division of a UnivariateDistribution with any other type.

        Types:
           - `UnivariateDistribution / UnivariateDistribution`: the result is the
             ratio of the vectorized versions of the two distributions.
           - `UnivariateDistribution / Any`: the left-parameter is vectorized and then
             divided by the right-parameter.

        """
        match other:
            case UnivariateDistribution():
                return self.to_vectorized() / other.to_vectorized()
            case _:
                return self.to_vectorized() / other

    def __pow__(self, other: Any) -> Any:
        """Return the left-power of a VectorDistribution with any other type.

        Types:
           - `UnivariateDistribution ** UnivariateDistribution`: the result is the
              vectorized version of the first distribution raised to the power
              of the two distributions.
           - `UnivariateDistribution ** Any`: the left-parameter is vectorized and then
             raised to the right-parameter.

        """
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
                return all(
                    isclose(value, vars(other)[field], abs_tol=Config.abs_tol)
                    for field, value in vars(self).items()
                )
            case _:
                return NotImplemented
