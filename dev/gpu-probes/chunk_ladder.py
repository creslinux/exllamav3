"""Chunk ladder 2048/4096/8192 at cap 2048 + relaxed guard. Error-checked (the arm-3
lesson: a reaped job looks like a clean fast exit), fresh seeds, per-shape warm,
rows-per-call trace. The error text (if any) names what bounds the 8192 shape."""
import os, time, random
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.generator import Generator, Job

config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=76800)
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False):
    pass
print("loaded", flush=True)
tokenizer = Tokenizer.from_config(config)

TPL = '''class Container{i}:
    """Entry {i}."""
    def __init__(self, id, capacity={c}):
        self.id = id; self.capacity = capacity; self.items = []
    def add(self, sku, qty=1):
        if len(self.items) + qty > self.capacity: return False
        self.items.extend([sku] * qty); return True
'''
V2 = '''def resolve_{i}(graph, root, depth={c}):
    seen, order = set(), []
    def visit(node, d):
        if node in seen or d > depth: return
        seen.add(node)
        for dep in graph.get(node, []): visit(dep, d + 1)
        order.append(node)
    visit(root, 0)
    return order
''', '''# metrics_{i}.py -- {c} samples per window
from dataclasses import dataclass
@dataclass
class S{i}:
    ts: float; value: float; w: float = 1.0
'''

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

def stats():
    torch.cuda.synchronize()
    calls, max_rows, over = 0, 0, 0
    for m in walk(model):
        rg = getattr(m, "_probe_region", None)
        if rg:
            calls += rg["entries"]; max_rows = max(max_rows, rg["max_rows"])
        st = getattr(m, "_probe_stats", None)
        if st: over += st["n_over"]
        for a in ("_probe_region", "_probe_stats", "_probe_tolist"):
            if hasattr(m, a): delattr(m, a)
    return calls, max_rows, over

for CH in (2048, 4096, 8192):
    gen = Generator(model, cache, tokenizer, max_chunk_size=CH)
    # per-shape warm (discarded)
    run_prefill(gen, tokenizer.encode("Warm.\n\n" + filler(170, 4000 + CH, True)))
    for tag, varied, seed in (("tpl", False, 5000 + CH), ("var", True, 6000 + CH)):
        ids = tokenizer.encode("Repo slice.\n\n" + filler(170, seed, varied))
        wall, errs = run_prefill(gen, ids)
        calls, max_rows, over = stats()
        if errs:
            print(f"ch{CH} {tag}: ERROR after {wall:.2f}s: {errs[0]}", flush=True)
        else:
            print(f"ch{CH} {tag}: {ids.shape[-1]} tok | wall {wall:.2f}s = {ids.shape[-1]/wall:.0f} T/s | "
                  f"count-path calls {calls} max_rows/call {max_rows} | overflow {over}", flush=True)
print("LADDER DONE", flush=True)
