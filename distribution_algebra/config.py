#!/usr/bin/env python3

import numpy as np


class Config:
    rng: np.random.Generator = np.random.default_rng(None)  # pyright: ignore
    sample_size: int = 100_000
    support_min: float = -20.0
    support_max: float = +20.0
    abs_tol: float = 1e-7
