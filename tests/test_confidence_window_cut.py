"""
Tests for Generator.confidence_window_cut — the batched post-loop form of the
historical sequential confidence break in the draft-model loops.

The reference below is an independent re-statement of the original in-loop
algorithm (per-step estimate, running reach product, break at idx + 1 when every
row sags below the calibrator confidence, guarded by idx + 1 < window). The
property test asserts the batched form reproduces it exactly on random inputs.

GPU-free.
"""

import os
import sys
import unittest
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.generator.generator import Generator


class StubCal:
    """
    Minimal calibrator standing in for DraftConfidence: estimate() maps a score
    through a fixed piecewise table, confidence is a scalar threshold.
    """
    def __init__(self, table, confidence):
        self.table = table
        self.confidence = confidence

    def estimate(self, score: float) -> float:
        for lo, val in self.table:
            if score >= lo:
                return val
        return self.table[-1][1]


def reference_cut(conf: torch.Tensor, cal, window: int) -> int:
    """
    Independent transcription of the pre-change in-loop break from
    iterate_draftmodel_gen / iterate_draftmodel_mtp_gen.
    """
    reach = None
    w = window
    for idx in range(window):
        est = [cal.estimate(v) for v in conf[:, idx].tolist()]
        reach = est if reach is None else [r * e for r, e in zip(reach, est)]
        if idx + 1 < window and max(reach) < cal.confidence:
            w = idx + 1
            break
    return w


def make_generator():
    g = Generator.__new__(Generator)
    return g


class ConfidenceWindowCutTest(unittest.TestCase):
    def test_matches_sequential_reference_on_random_matrices(self):
        rng = random.Random(1234)
        g = make_generator()
        for trial in range(500):
            rows = rng.randint(1, 8)
            window = rng.randint(1, 8)
            # random piecewise estimate table over [0, 2) score range
            n_bins = rng.randint(1, 6)
            edges = sorted(rng.uniform(0.0, 1.9) for _ in range(n_bins))
            table = [(edges[i], rng.uniform(0.05, 1.0)) for i in range(n_bins)]
            cal = StubCal(table, confidence = rng.uniform(0.05, 0.95))
            conf = torch.rand(rows, window) * 2.0
            got = g.confidence_window_cut(conf, cal, window)
            want = reference_cut(conf, cal, window)
            self.assertEqual(
                want, got,
                f"trial {trial}: table={table} conf={conf.tolist()} "
                f"confidence={cal.confidence}",
            )

    def test_no_cut_when_estimates_are_one(self):
        g = make_generator()
        cal = StubCal([(0.0, 1.0)], confidence = 0.9)  # burn-in behaviour
        conf = torch.zeros(3, 6)
        self.assertEqual(g.confidence_window_cut(conf, cal, 6), 6)

    def test_cut_at_first_position(self):
        g = make_generator()
        cal = StubCal([(0.0, 0.1)], confidence = 0.9)  # every estimate tiny
        conf = torch.ones(2, 6)
        self.assertEqual(g.confidence_window_cut(conf, cal, 6), 1)

    def test_single_position_window_never_cut(self):
        g = make_generator()
        cal = StubCal([(0.0, 0.01)], confidence = 0.9)
        conf = torch.zeros(1, 1)
        self.assertEqual(g.confidence_window_cut(conf, cal, 1), 1)


if __name__ == "__main__":
    unittest.main(verbosity = 2)
