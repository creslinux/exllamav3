"""A/B: TEMP_ROWS_FUSED 128 vs 1024 (env), two fillers, corrected collection."""
import os, time, random
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
os.environ["EXL3_PREFILL_PROBE"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.generator import Generator, Job

CAP = os.environ.get("EXL3_TEMP_ROWS_FUSED", "128")
config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=76800)
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False):
    pass
print(f"loaded (cap {CAP})", flush=True)
tokenizer = Tokenizer.from_config(config)
generator = Generator(model, cache, tokenizer)

TPL = '''class Container{i}:
    """Entry {i}."""
    def __init__(self, id, capacity={c}):
        self.id = id; self.capacity = capacity; self.items = []
    def add(self, sku, qty=1):
        if len(self.items) + qty > self.capacity: return False
        self.items.extend([sku] * qty); return True
'''
VAR = [
TPL,
'''def resolve_{i}(graph, root, depth={c}):
    seen, order = set(), []
    def visit(node, d):
        if node in seen or d > depth: return
        seen.add(node)
        for dep in graph.get(node, []): visit(dep, d + 1)
        order.append(node)
    visit(root, 0)
    return order
''',
'''# metrics_{i}.py -- {c} samples per window
from dataclasses import dataclass
@dataclass
class Sample_{i}:
    ts: float; value: float; weight: float = 1.0
def decay(samples, half_life=3600.0):
    if not samples: return 0.0
    return sum(s.value * s.weight for s in samples) / max(1e-9, sum(s.weight for s in samples))
''',
]

def filler(blocks, seed, varied):
    rng = random.Random(seed)
    if not varied:
        return "\n".join(TPL.format(i=rng.randrange(10000), c=rng.randrange(8, 4096)) for _ in range(blocks))
    out = []
    for i in range(blocks):
        t = VAR[i % len(VAR)]
        out.append(t.format(i=rng.randrange(10000), c=rng.randrange(8, 4096)))
    return "\n".join(out)

def walk(mod):
    for m in getattr(mod, "modules", []):
        yield m
        yield from walk(m)

def probe(blocks, seed, tag, varied):
    for m in walk(model):
        for a in ("_probe_stats", "_probe_region", "_probe_branches", "_probe_events", "_probe_tolist"):
            if hasattr(m, a): delattr(m, a)
    ids = tokenizer.encode("Repo slice.\n\n" + filler(blocks, seed, varied))
    job = Job(input_ids=ids, max_new_tokens=1)
    generator.enqueue(job)
    t0 = time.perf_counter()
    while generator.num_remaining_jobs():
        generator.iterate()
    wall = time.perf_counter() - t0
    torch.cuda.synchronize()
    tot = {"n_over": 0, "rows_over": 0, "max_c": 0, "layer_calls": 0, "loop_s": 0.0}
    tolist_s, ev_ms, region_calls = 0.0, 0.0, 0
    for m in walk(model):
        st = getattr(m, "_probe_stats", None)
        if st:
            for k in tot:
                tot[k] = max(tot[k], st[k]) if k == "max_c" else tot[k] + st[k]
        tolist_s += getattr(m, "_probe_tolist", 0.0)
        region_calls += getattr(m, "_probe_region", {"entries": 0})["entries"] if hasattr(m, "_probe_region") else 0
        evs = getattr(m, "_probe_events", None)
        if evs:
            for i in range(0, len(evs) - 1, 2):
                ev_ms += evs[i].elapsed_time(evs[i + 1])
    print(f"cap{CAP} {tag}: {ids.shape[-1]} tok | wall {wall:.2f}s = {ids.shape[-1]/wall:.0f} T/s | "
          f"tolist {tolist_s:.2f}s | region_gpu {ev_ms/1000:.2f}s | "
          f"overflow {tot['n_over']} experts (max single {tot['max_c']} rows) | loop_issue {tot['loop_s']:.2f}s", flush=True)

probe(170, 90, "warm   ", True)
probe(170, 2, "tpl~16k", False)
probe(678, 3, "tpl~64k", False)
probe(170, 12, "var~16k", True)
probe(678, 13, "var~64k", True)
print(f"AB DONE cap{CAP}", flush=True)
