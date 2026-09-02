"""
Per-block CUDA graph capture for decode (D3).

exllamav3 already captures each heavy submodule (GDN, MoE) internally, with
per-replay pointer patching -- but the ~10 eager ops around them per block
(hyper-connection mix/apply, norms, routing dispatch, dtype casts) are
launched from Python: ~500 launches per decoder pass, and the Step-0 profile
attributes ~22% + 10.5% of generation-thread time to the hyper-connection
and routing dispatch alone. This module captures the WHOLE TransformerBlock
forward (GDN-type blocks) as one torch.cuda.CUDAGraph, replayed per block
per pass.

Why torch.cuda.CUDAGraph works even though the internal BC graphs' replay is
illegal under capture (cudaError 900): during capture, the BC wrappers'
run_bszN calls are redirected to run_bszN_eager (a C++ binding added for
this purpose), which runs the layers' raw kernels via run_bszN_gr(..,
nullptr) -- plain stream launches, legal to record. The surrounding Python
(prepare_for_device, dtype checks, cached buffers, needs_configure) is host
code between kernel launches, which capture ignores.

Dynamic values per pass:
- x (the fp32 hyper-connection stream stack): copied into a static x_s per
  block before replay (small; blocks on a device can chain buffers in a
  later revision to remove these copies).
- recurrent-state slot indices: conv_state / recurrent_state are the
  layer-cache tensors themselves -- allocated once at load, address-stable
  across passes and jobs -- and the GDN kernels select rows via the
  device-side `slots` tensor at runtime (that is why the internal graphs
  pointer-patch it). A static slots_s per block is staged with the live
  indices each pass; batched across blocks with one _foreach_copy_.
- MoE routing: routing_std runs INSIDE the captured forward and writes the
  stable per-config routing buffers added earlier; the MoE kernels read
  them device-side. No staging needed.

Enabled with EXL3_BLOCK_GRAPH=1. Keys: (block, bsz, q_len, history).
Capture happens on the WARMUP'th invocation of a key, after submodule
autotune has settled. GDN blocks only -- QSA attention blocks contain
host-side branches (attn.py sparse threshold) and stay eager. The MTP draft
head is attention-type and also stays eager.
"""
from __future__ import annotations
import os
import torch

ENABLED = os.environ.get("EXL3_BLOCK_GRAPH", "0").lower() not in ("", "0", "false", "no")
WARMUP = int(os.environ.get("EXL3_BLOCK_GRAPH_WARMUP", "3"))

_in_capture = False
_warmup: dict = {}
_graphs: dict = {}
_slots_stage: dict = {}       # key -> (static_slots, live_slots_ref)
_stage_in: list = []
_stage_out: list = []


class _ShimmedBC:
    """
    Redirects run_bszN to the eager path while a capture is active.
    """
    def __init__(self, bc):
        self._bc = bc

    def __getattr__(self, name):
        return getattr(self._bc, name)

    def run_bszN(self, *args):
        if _in_capture:
            return self._bc.run_bszN_eager(*args)
        return self._bc.run_bszN(*args)


def eligible(block, params: dict, x: torch.Tensor) -> bool:
    if not ENABLED:
        return False
    attn = getattr(block, "attn", None)
    if attn is None or not getattr(attn, "bc_split", False) or getattr(attn, "bc", None) is None:
        return False
    if params.get("prefill") or params.get("export_state_layers"):
        return False
    bsz, qlen = x.size(0), x.size(1)
    # True decode shapes only: single-token decode/draft (q_len 1) and MTP
    # verify windows (q_len = num_draft + 1). Larger q_len covers unflagged
    # prefill tail-chunks etc., whose MoE path (bincount/tolist) contains
    # capture-illegal host syncs.
    if qlen > 7:
        return False
    if bsz * qlen > 8:
        return False
    return True


def _live_slots(params, device):
    return params.get("recurrent_slots")


def _shim_block(block):
    """
    Install the capture-redirecting shims on the block's attn/mlp BC objects.
    Idempotent.
    """
    for mod in (getattr(block, "attn", None), getattr(block, "mlp", None)):
        bc = getattr(mod, "bc", None) if mod is not None else None
        if bc is not None and not isinstance(bc, _ShimmedBC):
            mod.bc = _ShimmedBC(bc)


def run(block, x: torch.Tensor, params: dict):
    """
    Entry point replacing block.forward for eligible decode forwards.
    """
    global _in_capture
    key = (id(block), x.size(0), x.size(1), bool(params.get("recurrent_history")))
    entry = _graphs.get(key)

    if entry is None:
        n = _warmup.get(key, 0) + 1
        _warmup[key] = n
        if n < WARMUP:
            return block.forward(x, params)

        # Capture. The slot-index tensor is swapped for a static buffer so
        # replay-time indices can be staged without touching the graph; the
        # state cache tensors themselves are address-stable and pass through.
        device = x.device
        slots_live = _live_slots(params, device)
        # params["recurrent_slots"] may be a CPU tensor uploaded lazily via the
        # params dev-cache; the static copy must live on this block's device
        slots_s = slots_live.detach().to(device, copy = True)
        params_c = dict(params)
        # get_for_device caches by tensor identity in params["dev_cache"];
        # a fresh dict without the cache ensures our static buffer is what
        # the GDN layers read.
        params_c.pop("dev_cache", None)
        params_c["recurrent_slots"] = slots_s

        graph = torch.cuda.CUDAGraph()
        x_s = x.clone()
        _shim_block(block)
        side = torch.cuda.Stream(device = device)
        side.wait_stream(torch.cuda.current_stream(device))
        # Re-entrancy guard: the settle and capture calls below must run the
        # eager body (TransformerBlock.forward's gate checks _in_capture), or
        # each capture attempt would re-enter run() and recurse
        prev = _in_capture
        _in_capture = True
        try:
            with torch.cuda.stream(side):
                _ = block.forward(x_s, params_c)      # settle lazy state
            torch.cuda.current_stream(device).wait_stream(side)
            with torch.cuda.graph(graph, stream = side):
                y_s = block.forward(x_s, params_c)
        finally:
            _in_capture = prev
        _graphs[key] = (graph, x_s, y_s, slots_s)
        return y_s

    graph, x_s, y_s, slots_s = entry
    slots_live = _live_slots(params, x.device)
    if slots_live.data_ptr() != slots_s.data_ptr():
        slots_s.copy_(slots_live, non_blocking = True)
    if x.data_ptr() != x_s.data_ptr():
        x_s.copy_(x, non_blocking = True)
    graph.replay()
    return y_s
