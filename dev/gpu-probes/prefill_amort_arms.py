"""Three-arm dequant-amortisation rig. Env: EXL3_TEMP_ROWS_FUSED; argv[1]: chunk size."""
import os, sys, time, random
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
os.environ["EXL3_PREFILL_PROBE"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.generator import Generator, Job

CAP = os.environ.get("EXL3_TEMP_ROWS_FUSED", "128")
CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=76800)
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False):
    pass
print(f"loaded (cap {CAP}, chunk {CHUNK})", flush=True)
tokenizer = Tokenizer.from_config(config)
generator = Generator(model, cache, tokenizer, max_chunk_size=CHUNK)

TPL = '''class Container{i}:
    """Entry {i}."""
    def __init__(self, id, capacity={c}):
        self.id = id; self.capacity = capacity; self.items = []
    def add(self, sku, qty=1):
        if len(self.items) + qty > self.capacity: return False
        self.items.extend([sku] * qty); return True
'''
VAR2 = '''def resolve_{i}(graph, root, depth={c}):
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
        out.append([TPL, VAR2[0], VAR2[1]][i % 3].format(i=rng.randrange(10000), c=rng.randrange(8, 4096)))
    return "\n".join(out)

def walk(mod):
    for m in getattr(mod, "modules", []):
        yield m
        yield from walk(m)

def probe(blocks, seed, tag, varied):
    for m in walk(model):
        for a in ("_probe_stats", "_probe_region", "_probe_events", "_probe_tolist", "_probe_hist"):
            if hasattr(m, a): delattr(m, a)
    ids = tokenizer.encode("Repo slice.\n\n" + filler(blocks, seed, varied))
    job = Job(input_ids=ids, max_new_tokens=1)
    generator.enqueue(job)
    t0 = time.perf_counter()
    while generator.num_remaining_jobs():
        generator.iterate()
    wall = time.perf_counter() - t0
    torch.cuda.synchronize()
    tot = {"n_over": 0, "rows_over": 0, "layer_calls": 0, "loop_s": 0.0}
    hottest, tolist_s, ev_ms, distinct, chunks = 0, 0.0, 0.0, 0, 0
    for m in walk(model):
        st = getattr(m, "_probe_stats", None)
        if st:
            tot["n_over"] += st["n_over"]; tot["rows_over"] += st["rows_over"]
            tot["layer_calls"] += st["layer_calls"]; tot["loop_s"] += st["loop_s"]
        h = getattr(m, "_probe_hist", None)
        if h:
            distinct += h["distinct"]; chunks += h["chunks"]
            hottest = max(hottest, h["hottest"])
        tolist_s += getattr(m, "_probe_tolist", 0.0)
        evs = getattr(m, "_probe_events", None)
        if evs:
            for i in range(0, len(evs) - 1, 2):
                ev_ms += evs[i].elapsed_time(evs[i + 1])
    print(f"cap{CAP}/ch{CHUNK} {tag}: {ids.shape[-1]} tok | wall {wall:.2f}s = {ids.shape[-1]/wall:.0f} T/s | "
          f"tolist {tolist_s:.2f}s | region_gpu {ev_ms/1000:.2f}s | chunks {chunks//48} | "
          f"distinct/lc {distinct//max(1,chunks)} | hottest {hottest} | over {tot['n_over']} exp "
          f"({tot['rows_over']} rows) | loop {tot['loop_s']:.2f}s", flush=True)

probe(170, 90, "warm   ", True)
probe(170, 2, "tpl~16k", False)
probe(170, 12, "var~16k", True)
print(f"ARM DONE cap{CAP}/ch{CHUNK}", flush=True)
