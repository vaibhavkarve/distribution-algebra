#!/usr/bin/env python3


from distribution_algebra.lognormal import Lognormal
from distribution_algebra.normal import Normal


def example_addition() -> None:
    a: Normal = Normal(mean=1.0, var=9.0)
    b: Normal = Normal(mean=2.0, var=16.0)

    c: Normal = Normal(mean=3.0, var=25.0)
    assert c == a + b


def example_multiplication() -> None:
    a: Lognormal = Lognormal(mean=1.0, var=1.0)
    b: Lognormal = Lognormal(mean=2.0, var=1.0)

    c_normal_mean: float = a.normal_mean + b.normal_mean
    c_normal_var: float = a.normal_var + b.normal_var
    c: Lognormal = Lognormal.from_normal_mean_var(c_normal_mean, c_normal_var)
    assert c == a * b


if __name__ == '__main__':
    example_addition()
    example_multiplication()
