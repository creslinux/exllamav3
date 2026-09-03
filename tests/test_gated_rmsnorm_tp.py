"""
Tests for the GatedRMSNorm TP round-trip: gate_activation must survive
tp_export -> tp_import (and _split). qwen4_exp is the only architecture that
reads output_gate_type from the model config (sigmoid); under TP the BC
constructor was called without the activation selector, silently rebuilding
the norm with the silu default.

GPU-free: the BC constructor is mocked.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.modules.gated_rmsnorm import GatedRMSNorm


class _StubProducer:
    def send(self, t):
        return t


class _StubConsumer:
    def recv(self, t, cuda=None):
        return t


def make_norm(gate_activation):
    m = GatedRMSNorm.__new__(GatedRMSNorm)
    m.key = "test.norm"
    m.rms_norm_eps = 1e-6
    m.out_dtype = torch.half
    m.constant_bias = 0.0
    m.groups = 1
    m.gate_first = True
    m.gate_activation = gate_activation
    m.device = torch.device("cuda:0")
    m.weight = torch.ones(8, dtype = torch.half)
    return m


class GatedRMSNormTptest(unittest.TestCase):
    def _run(self, act):
        captured = {}

        def fake_bc(weight, eps, bias, groups, gate_first, gate_sel = None):
            captured["args"] = (eps, bias, groups, gate_first, gate_sel)
            return object()

        m = make_norm(act)
        exported = m.tp_export(None, _StubProducer())
        self.assertEqual(exported["kwargs"]["gate_activation"], act)

        local = {"consumer": _StubConsumer(), "device": torch.device("cuda:0")}
        with patch("exllamav3.modules.gated_rmsnorm.ext.BC_GatedRMSNorm", fake_bc), \
             patch("exllamav3.modules.gated_rmsnorm.torch.cuda.synchronize"):
            GatedRMSNorm.tp_import(local, exported, None)
        eps, bias, groups, gate_first, sel = captured["args"]
        self.assertEqual(sel, 1 if act == "sigmoid" else 0, f"{act}: BC selector")

        # split variant
        with patch("exllamav3.modules.gated_rmsnorm.ext.BC_GatedRMSNorm", fake_bc), \
             patch("exllamav3.modules.gated_rmsnorm.torch.cuda.synchronize"):
            GatedRMSNorm.tp_import_split(local, exported, None, (0, 8))
        eps, bias, groups, gate_first, sel = captured["args"]
        self.assertEqual(sel, 1 if act == "sigmoid" else 0, f"{act}: split BC selector")

    def test_sigmoid_survives_roundtrip(self):
        self._run("sigmoid")

    def test_silu_survives_roundtrip(self):
        self._run("silu")


if __name__ == "__main__":
    unittest.main(verbosity = 2)
