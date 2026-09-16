"""Global RNG seeding for reproducible runs.

Known residual non-determinism (e.g. gensim ``workers > 1``, non-deterministic
GAT kernels) is not silently hidden here — callers declare it explicitly in the
run manifest's ``env.deterministic`` field.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy``, ``torch`` (CPU + CUDA), and ``PYTHONHASHSEED``.

    ``PYTHONHASHSEED`` only affects hash randomization for subprocesses started
    after this call; it is set here for completeness and manifest transparency,
    not because it changes behavior within the current process.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
