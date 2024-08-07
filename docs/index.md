**A python package that implements an easy-to-use interface for random
variables, statistical distributions, and their algebra.**

<a
href="https://img.shields.io/github/license/vaibhavkarve/distribution-algebra?style=flat-square">
<img
src="https://img.shields.io/github/license/vaibhavkarve/distribution-algebra?style=flat-square"
alt="License"> </a> <a
href="https://img.shields.io/badge/Python-v3.10-blue?style=flat-square">
<img
src="https://img.shields.io/badge/Python-v3.10-blue?style=flat-square"
alt="Python:v3.10"> </a>

This Python package is brought to you by [Vaibhav
Karve](https://vaibhavkarve.github.io).

`distribution-algebra` recognizes Normal, Lognormal, Beta and Poisson
distributions. The package implements an interface to easily construct
user-defined Univariate distributions as well Vectorized
distributions.

Additional features include:
- A `plot` function for probability density/mass function plotting.
- A `draw` function for drawing random samples of specified size from
  a given distribution.
- Addition and multiplication operations defined directly on
  distributions:
  - For example, the sum of two Normal (Poisson) distributions is
    Normal (Poisson).
  - The product of two Lognormal distributions is Lognormal.
  - The sum of two arbitrary univariate distributions is expressed as
    a Vectorized distribution.


![Example
plot](https://github.com/vaibhavkarve/distribution-algebra/raw/main/docs/example_plot_univariate.png)


This package is written in Python v3.10, and is available for
installation from
[PyPI](https://pypi.org/project/distribution-algebra/) under the
[GNU-GPL-v3.0
license](https://github.com/vaibhavkarve/normal-form/blob/main/LICENSE).


# Installation and usage

To get started on using this package,

1.  Install Python 3.10 or higher.
2.  `python3.10 -m pip install distribution-algebra`
3.  Use it in a python script (or interactive REPL)

```python title="Example showing addition of two Normal distributions"
from distribution_algebra import Normal  # (1)!

x: Normal = Normal(mean=1.0, var=9.0)
y: Normal = Normal(mean=1.0, var=16.0)

assert x + y == Normal(mean=2.0, var=25.0)
```

1. You can similarly import other distributions as `from distribution_algebra import Normal, Beta, Lognormal, Beta4, Poisson`.
