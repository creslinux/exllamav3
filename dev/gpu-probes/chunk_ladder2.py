"""Ladder rerun: cap 2048 + guard, chunks 2048 vs 4096, SAME seeds, probe env ON.
Verifies the chunk actually grew (4096 > cap => count path must fire)."""
import os, time, random
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
os.environ["EXL3_PREFILL_PROBE"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.generator import Generator, Job

config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=76800)
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False): pass
print("loaded", flush=True)
tokenizer = Tokenizer.from_config(config)

TPL = 'class Container{i}:\n    """Entry {i}."""\n    def __init__(self, id, capacity={c}):\n        self.id = id; self.capacity = capacity; self.items = []\n    def add(self, sku, qty=1):\n        if len(self.items) + qty > self.capacity: return False\n        self.items.extend([sku] * qty); return True\n'
V2 = ('def resolve_{i}(graph, root, depth={c}):\n    seen, order = set(), []\n    def visit(node, d):\n        if node in seen or d > depth: return\n        seen.add(node)\n        for dep in graph.get(node, []): visit(dep, d + 1)\n        order.append(node)\n    visit(root, 0)\n    return order\n',
     '# metrics_{i}.py -- {c} samples per window\nfrom dataclasses import dataclass\n@dataclass\nclass S{i}:\n    ts: float; value: float; w: float = 1.0\n')

def filler(blocks, seed, varied):
    rng = random.Random(seed)
    if not varied:
        return "\n".join(TPL.format(i=rng.randrange(10000), c=rng.randrange(8, 4096)) for _ in range(blocks))
    out = []
    for i in range(blocks):
        out.append([TPL, V2[0], V2[1]][i % 3].format(i=rng.randrange(10000), c=rng.randrange(8, 4096)))
    return "\n".join(out)

def walk(mod):
    for m in getattr(mod, "modules", []):
        yield m
        yield from walk(m)

def run_prefill(gen, ids):
    job = Job(input_ids=ids, max_new_tokens=1)
    gen.enqueue(job)
    t0 = time.perf_counter()
    errors = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if isinstance(r, dict) and r.get("error"):
                errors.append(str(r.get("error"))[:400])
    return time.perf_counter() - t0, errors

def clear():
    for m in walk(model):
        for a in ("_probe_region", "_probe_stats", "_probe_tolist", "_probe_hist"):
            if hasattr(m, a): delattr(m, a)

def region_stats():
    torch.cuda.synchronize()
    calls, max_rows, total_rows = 0, 0, 0
    for m in walk(model):
        rg = getattr(m, "_probe_region", None)
        if rg:
            calls += rg["entries"]; max_rows = max(max_rows, rg["max_rows"]); total_rows += rg["rows"]
    return calls, max_rows, total_rows

# same seeds for both chunk sizes
for CH in (2048, 4096):
    gen = Generator(model, cache, tokenizer, max_chunk_size=CH)
    # warm at this shape (seed same for warm)
    run_prefill(gen, tokenizer.encode("Warm.\n\n" + filler(170, 7777, True)))
    for tag, varied, seed in (("tpl", False, 3333), ("var", True, 4444)):
        clear()
        ids = tokenizer.encode("Repo slice.\n\n" + filler(170, seed, varied))
        wall, errs = run_prefill(gen, ids)
        calls, max_rows, total_rows = region_stats()
        if errs:
            print(f"ch{CH} {tag}: ERROR: {errs[0]}", flush=True)
        else:
            print(f"ch{CH} {tag}: {ids.shape[-1]} tok | wall {wall:.2f}s = {ids.shape[-1]/wall:.0f} T/s | "
                  f"count-path calls {calls} max_rows/call {max_rows} total_rows {total_rows}", flush=True)
print("LADDER2 DONE", flush=True)
