
#!/usr/bin/env python3

from typing import Final

import numpy as np

RNG: Final[np.random.Generator] = np.random.default_rng(None)  # pyright: ignore
SAMPLE_SIZE: Final[int] = 100_000
SUPPORT_MIN: Final[float] = -20.0
SUPPORT_MAX: Final[float] = +20.0
ABS_TOL: Final[float] = 1e-7
