"""Tests for seed_everything reproducibility."""

from __future__ import annotations

import random

import numpy as np

from cga.core.seeds import seed_everything


def test_seed_everything_reproduces_random_and_numpy_sequences():
    seed_everything(123)
    py_seq_1 = [random.random() for _ in range(5)]
    np_seq_1 = np.random.rand(5)

    seed_everything(123)
    py_seq_2 = [random.random() for _ in range(5)]
    np_seq_2 = np.random.rand(5)

    assert py_seq_1 == py_seq_2
    assert np.array_equal(np_seq_1, np_seq_2)


def test_seed_everything_different_seeds_differ():
    seed_everything(1)
    a = random.random()
    seed_everything(2)
    b = random.random()
    assert a != b
