#!/usr/bin/env python3

import numpy as np
from attr import define
from typing import ClassVar


@define
class Config:
    rng: ClassVar[np.random.Generator] = np.random.default_rng(None)  # pyright: ignore
    sample_size: ClassVar[int] = 100_000
    support_min: ClassVar[float] = -20.0
    support_max: ClassVar[float] = +20.0
    abs_tol: ClassVar[float] = 1e-7
