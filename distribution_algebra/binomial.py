#!/usr/bin/env python3

from dataclasses import field
from math import floor, inf
from typing import Any
from numbers import Real

import numpy as np
import scipy
from attr import field, frozen, validators
from numpy.typing import NDArray

from distribution_algebra.config import RNG
from distribution_algebra.distribution import UnivariateDistribution


@frozen(kw_only=True)
class Binomial(UnivariateDistribution[np.int64]):
    # Total number of trials with yes-or-no questions.
    n: int = field(validator=validators.ge(0))
    # Probability of yes/success outcome.
    p: float = field(validator=[validators.ge(0.0), validators.le(1.0)])

    def draw(self, size: int) -> NDArray[np.int_]:
        return RNG.binomial(n=self.n, p=self.p, size=size)

    @property
    def mean(self) -> float:
        return self.n * self.p

    @property
    def var(self) -> float:
        return self.n * self.p * (1 - self.p)

    @property
    def median(self) -> float:
        if self.p == 0.5 and self.n % 2:
            return floor((self.n + 1) // 2)
        if self.p == 0.5 and not (self.n % 2):
            return self.n // 2
        return round(self.n * self.p)

    @property
    def mode(self) -> int:
        np_plus_p: float = (self.n + 1) * self.p
        match np_plus_p:
            case 0:
                assert not self.p
                return 0
            case _ if np_plus_p == self.n + 1:
                assert self.p == 1
                return self.n
            case _ if np_plus_p.is_integer():
                return int(np_plus_p)
            case _ if not np_plus_p.is_integer():
                return floor(np_plus_p)
            case _:
                raise RuntimeError("{self.n = }, {self.p = } slipped through the match-case.")

    @property
    def support(self) -> tuple[int, float]:
        return 0, inf

    def pdf(self, arange: NDArray[np.int_]) -> NDArray[np.float64]:  # type: ignore
        return scipy.stats.binom.pmf(  # type: ignore
            k=arange, n=self.n, p=self.p)

    def __add__(self, other: Any) -> Any:
        match other:
            case Binomial(n=n2, p=p2) if self.p == p2:
                return Binomial(n=self.n + n2, p=self.p)  # type: ignore
            case _:
                return super().__add__(other)
