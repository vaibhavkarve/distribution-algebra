#!/usr/bin/env python3

from math import floor
from typing import Any

import attr
import numpy as np
import scipy
from numpy.typing import NDArray

from distribution_algebra.config import Config
from distribution_algebra.distribution import UnivariateDistribution


@attr.frozen(kw_only=True)
class Binomial(UnivariateDistribution[np.int_]):
    # Total number of trials with yes-or-no questions.
    n: int = attr.field(validator=attr.validators.ge(0))
    # Probability of yes/success outcome.
    p: float = attr.field(validator=[attr.validators.ge(0.0), attr.validators.le(1.0)])

    def draw(self, size: int) -> NDArray[np.int_]:
        return Config.rng.binomial(n=self.n, p=self.p, size=size)

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
    def mode(self) -> int:  # type: ignore[return]
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
                raise RuntimeError(
                    "{self.n = }, {self.p = } slipped through the match-case."
                )

    @property
    def support(self) -> tuple[np.int_, np.int_]:
        return np.int_(0), np.int_(np.iinfo(np.int_).max)  # max = np.inf.

    def pdf(self, arange: NDArray[np.int_]) -> NDArray[np.float64]:  # type: ignore
        return scipy.stats.binom.pmf(k=arange, n=self.n, p=self.p)  # type: ignore

    def __add__(self, other: Any) -> Any:
        match other:
            case Binomial(n=n2, p=p2) if self.p == p2:
                return Binomial(n=self.n + n2, p=self.p)  # type: ignore
            case _:
                return super().__add__(other)
