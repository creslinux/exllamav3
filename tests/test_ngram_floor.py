"""
Tests for two-stage speculation's n-gram floor logic in iterate_ngram_gen.

GPU-free: stub jobs provide fixed n-gram drafts; the selection contract is
(1) every job at/above the floor -> concatenated draft trimmed to the batch
minimum, (2) any job below the floor -> None (fall back to the MTP drafter),
(3) floor None preserves the original behaviour (only min_len == 0 returns
None).
"""

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.generator.generator import Generator


class StubJob:
    def __init__(self, draft_len):
        self.draft_len = draft_len

    def is_prefill_done(self):
        return True

    def get_max_seq_len(self):
        return 16

    def get_ngram_draft(self, window):
        return torch.full((1, min(self.draft_len, window)), 7, dtype = torch.long)


def make_generator(jobs):
    g = Generator.__new__(Generator)
    g.active_jobs = jobs
    g.num_draft_tokens = 6
    return g


class NgramFloorTest(unittest.TestCase):
    def test_all_above_floor_returns_draft(self):
        g = make_generator([StubJob(6), StubJob(4)])
        out = g.iterate_ngram_gen([], min_len_floor = 3)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (2, 4))  # trimmed to batch min
        self.assertTrue((out == 7).all())

    def test_any_below_floor_returns_none(self):
        g = make_generator([StubJob(6), StubJob(2)])
        self.assertIsNone(g.iterate_ngram_gen([], min_len_floor = 3))

    def test_floor_exact_boundary(self):
        g = make_generator([StubJob(3)])
        out = g.iterate_ngram_gen([], min_len_floor = 3)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (1, 3))

    def test_no_floor_preserves_original(self):
        g = make_generator([StubJob(6), StubJob(2)])
        out = g.iterate_ngram_gen([])  # floor None -> legacy behaviour
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (2, 2))

        g = make_generator([StubJob(0)])
        self.assertIsNone(g.iterate_ngram_gen([]))


if __name__ == "__main__":
    unittest.main(verbosity = 2)
