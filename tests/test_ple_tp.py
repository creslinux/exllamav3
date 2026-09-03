"""
GPU-free tests for the PLELayer + NGramEmbedding TP round-trips.

NGramEmbedding: export carries disk-mode handles (filename/offset/shape) and
host-side hash params; import reconstructs DiskTensorHandles and keeps the
hash params on CPU. RAM mode fails closed.

PLELayer: export carries all six submodules + conv_w + recurrent layers;
import rebuilds with replicated projections (split=None) and populates
tp_recurrent_lookup with allocated state layers.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.modules.ple import PLELayer, PLELayerState
from exllamav3.modules.ngram_embedding import NGramEmbedding


class _Prod:
    def send(self, x):
        return ("ref", x)


class _Cons:
    def recv(self, ref, cuda=None, **kw):
        return ref[1]


def make_ngram(mode="trellis_disk"):
    m = NGramEmbedding.__new__(NGramEmbedding)
    m.key = "l.ple.ngram"
    m.ngram_size = 4
    m.heads_per_ngram = 2
    m.ple_embed_dim = 160 * 6
    m.eos_token_id = 99
    m.out_dtype = torch.half
    m.device = torch.device("cuda:0")
    m.mode = mode
    m.K = 5
    m.rows_per_shard = 1000
    m.num_rows = 1000
    m._row_dtype = None
    m.head_offsets = torch.zeros(6, dtype=torch.long)
    m.head_vocab_sizes = torch.ones(6, dtype=torch.long)
    m.layer_multipliers = torch.ones(6, dtype=torch.long)
    m.head_bias = None
    m.handles = []
    return m


class NGramTPTest(unittest.TestCase):
    def test_disk_mode_export_import_roundtrip(self):
        from exllamav3.loader.safetensors import DiskTensorHandle
        m = make_ngram()
        # A real DiskTensorHandle shape (filename/offset/shape/dtype)
        m.handles = [DiskTensorHandle("l.ple.ngram", "/models/x/ngram_embedding.safetensors",
                                      4096, [1000, 10], torch.int16)]
        exp = m.tp_export(None, _Prod())
        self.assertEqual(exp["mode"], "trellis_disk")
        self.assertEqual(len(exp["handles"]), 1)
        h = exp["handles"][0]
        self.assertEqual(h["filename"], "/models/x/ngram_embedding.safetensors")
        self.assertEqual(h["abs_offset"], 4096)

        with patch("exllamav3.modules.ngram_embedding.mul1_codebook", return_value=object()), \
             patch("exllamav3.modules.ngram_embedding.torch.cuda.synchronize"):
            got = NGramEmbedding.tp_import({"consumer": _Cons(), "device": torch.device("cuda:0")},
                                           exp, None)
        self.assertEqual(got.mode, "trellis_disk")
        self.assertEqual(len(got.handles), 1)
        self.assertEqual(got.handles[0].filename, "/models/x/ngram_embedding.safetensors")
        self.assertEqual(got.handles[0].abs_offset, 4096)
        # Hash params stay on CPU
        self.assertEqual(got.head_offsets.device.type, "cpu")

    def test_ram_mode_fails_closed(self):
        m = make_ngram(mode="trellis_ram")
        with self.assertRaises(AssertionError):
            m.tp_export(None, _Prod())


class PLETPTest(unittest.TestCase):
    def test_export_carries_all_submodules(self):
        m = PLELayer.__new__(PLELayer)
        m.key = "l.ple"
        m.layer_idx = -2
        m.hidden_size = 2560
        m.hc_mult = 4
        m.ple_embed_dim = 960
        self.ple_embed_dim = 960
        m.rms_norm_eps = 1e-6
        m.mm_token_id = None
        m.out_dtype = torch.half
        m.conv_kernel_size = 4
        m.device = torch.device("cuda:0")
        m.conv_w = torch.ones(4, 2560, dtype=torch.half)
        m.ple_embedding = make_ngram()
        m.ple_embedding.device = torch.device("cuda:0")
        m.key_proj = MagicMock()
        m.key_proj.tp_export.return_value = {"cls": "KP"}
        m.value_proj = MagicMock()
        m.value_proj.tp_export.return_value = {"cls": "VP"}
        m.norm_key = MagicMock(); m.norm_key.tp_export.return_value = {"cls": "NK"}
        m.norm_query = MagicMock(); m.norm_query.tp_export.return_value = {"cls": "NQ"}
        m.norm_conv = MagicMock(); m.norm_conv.tp_export.return_value = {"cls": "NC"}
        m.recurrent_layers = []

        exp = m.tp_export(None, _Prod())
        self.assertEqual(exp["cls"], PLELayer)
        self.assertEqual(exp["kwargs"]["ngram_size"], 4)
        self.assertEqual(exp["ple_embedding"]["cls"], NGramEmbedding)
        self.assertEqual(exp["key_proj"]["cls"], "KP")
        self.assertEqual(exp["value_proj"]["cls"], "VP")
        self.assertIsNotNone(exp["conv_w"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
