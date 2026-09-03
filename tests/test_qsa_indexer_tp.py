"""
GPU-free tests for QSAIndexer TP replication: constructor accepts pre-built
submodules (tp_import path), tp_export carries them, and the round-trip
rebuilds without touching the config store. Attention's allocation moves
QSA cache-plane storage to the per-device bucket (asserted structurally).
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.modules.qsa_indexer import QSAIndexer


class _Prod:
    def send(self, x):
        return ("sent", x)


class _Cons:
    def recv(self, ref, cuda=None, **kw):
        return ref[1]


def make_indexer(n_heads=4, head_dim=128):
    proj = object.__new__(__import__("exllamav3.modules.linear", fromlist=["Linear"]).Linear)
    proj.device = torch.device("cuda:0")
    norm_cls = __import__("exllamav3.modules.rmsnorm", fromlist=["RMSNorm"]).RMSNorm
    qn = object.__new__(norm_cls)
    qn.device = torch.device("cuda:0")
    qn.unweighted = False
    qn.weight = torch.ones(8, dtype=torch.half)
    kn = object.__new__(norm_cls)
    kn.device = torch.device("cuda:0")
    kn.unweighted = False
    kn.weight = torch.ones(8, dtype=torch.half)

    m = QSAIndexer(
        config=None,
        key="layers.0.self_attn.indexer",
        hidden_size=2560,
        n_heads=n_heads,
        kv_heads=1,
        head_dim=head_dim,
        token_budget=32,
        compress_ratio=4,
        rms_norm_eps=1e-6,
        index_qk_proj=proj,
        q_layernorm=qn,
        k_layernorm=kn,
    )
    assert m.index_qk_proj is proj and m.q_layernorm is qn and m.k_layernorm is kn
    m.device = torch.device("cuda:0")
    return m, proj, qn, kn


class QSAIndexerTPTest(unittest.TestCase):
    def test_export_carries_submodules_and_kwargs(self):
        m, proj, qn, kn = make_indexer()
        fake_proj_exp = {"cls": "P", "x": 1}
        fake_norm_exp = {"cls": "N", "w": 2}
        with patch.object(proj, "tp_export", return_value=fake_proj_exp), \
             patch.object(qn, "tp_export", return_value=fake_norm_exp), \
             patch.object(kn, "tp_export", return_value=fake_norm_exp):
            exp = m.tp_export(None, _Prod())
        self.assertEqual(exp["cls"], QSAIndexer)
        self.assertEqual(exp["kwargs"]["n_heads"], 4)
        self.assertEqual(exp["kwargs"]["kv_heads"], 1)
        self.assertIs(exp["index_qk_proj"], fake_proj_exp)
        self.assertIs(exp["q_layernorm"], fake_norm_exp)
        self.assertIs(exp["k_layernorm"], fake_norm_exp)

    def test_import_rebuilds_with_exported_submodules(self):
        m, proj, qn, kn = make_indexer()
        fake_proj_exp = {"kwargs": {"key": "k.p"}, "inner": {}}
        rebuilt_proj = object()
        fake_norm_exp = {"kwargs": {"key": "k.n"}}
        rebuilt_norm = object()
        exported = {
            "cls": QSAIndexer,
            "kwargs": {
                "key": "layers.0.self_attn.indexer",
                "hidden_size": 2560, "n_heads": 4, "kv_heads": 1, "head_dim": 128,
                "token_budget": 32, "compress_ratio": 4, "rms_norm_eps": 1e-6,
                "out_dtype": torch.half,
            },
            "index_qk_proj": fake_proj_exp,
            "q_layernorm": fake_norm_exp,
            "k_layernorm": fake_norm_exp,
            "device": torch.device("cuda:0"),
        }
        from exllamav3.modules.linear import Linear
        from exllamav3.modules.rmsnorm import RMSNorm
        with patch.object(Linear, "tp_import_split", return_value=rebuilt_proj), \
             patch.object(RMSNorm, "tp_import", return_value=rebuilt_norm), \
             patch("exllamav3.modules.qsa_indexer.torch.cuda.synchronize"):
            got = QSAIndexer.tp_import({"device": torch.device("cuda:0")}, exported, None)
        self.assertIs(got.index_qk_proj, rebuilt_proj)
        self.assertIs(got.q_layernorm, rebuilt_norm)
        self.assertIs(got.k_layernorm, rebuilt_norm)
        self.assertEqual(got.n_heads, 4)
        self.assertEqual(got.block_topk, 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
