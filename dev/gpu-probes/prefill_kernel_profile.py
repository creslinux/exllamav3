"""Per-kernel prefill profile: fused MoE kernel duration vs routing histogram, both fillers.
Also yields the full per-class split (GDN/QSA/MoE/glue) from the same run."""
import os, time, random
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
os.environ["EXL3_PREFILL_PROBE"] = "1"
import torch
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.generator import Generator, Job

config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=76800)
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False):
    pass
print("loaded", flush=True)
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
        t = [TPL, VAR2[0], VAR2[1]][i % 3]
        out.append(t.format(i=rng.randrange(10000), c=rng.randrange(8, 4096)))
    return "\n".join(out)

def walk(mod):
    for m in getattr(mod, "modules", []):
        yield m
        yield from walk(m)

def reset():
    for m in walk(model):
        for a in ("_probe_hist", "_probe_tolist", "_probe_events", "_probe_stats", "_probe_region"):
            if hasattr(m, a): delattr(m, a)

def run_prefill(ids):
    job = Job(input_ids=ids, max_new_tokens=1)
    generator.enqueue(job)
    t0 = time.perf_counter()
    while generator.num_remaining_jobs():
        generator.iterate()
    return time.perf_counter() - t0

def report(tag, prof):
    agg = {}
    for ev in prof.key_averages():
        if ev.device_type == torch.autograd.DeviceType.CUDA or (ev.self_device_time_total or 0) > 0:
            agg[ev.key] = (ev.self_device_time_total / 1000.0, ev.count)
    top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:14]
    hist = {"distinct": 0, "hottest": 0, "chunks": 0}
    for m in walk(model):
        h = getattr(m, "_probe_hist", None)
        if h:
            for k in hist: hist[k] += h[k]
    n_layers = 48
    chunks = hist["chunks"] / n_layers
    avg_distinct = hist["distinct"] / max(1, hist["chunks"])
    moe_ms = sum(v[0] for k, v in agg.items() if "exl3_moe" in k)
    moe_calls = sum(v[1] for k, v in agg.items() if "exl3_moe" in k)
    print(f"\n== {tag}: chunks~{chunks:.0f} | distinct experts/layer-chunk {avg_distinct:.0f} | hottest {hist['hottest']} rows", flush=True)
    print(f"   fused moe kernel: {moe_ms:.1f} ms total, {moe_calls} calls = {moe_ms/max(1,hist['chunks']):.2f} ms/layer-chunk", flush=True)
    for k, (ms, n) in top:
        print(f"   {ms:9.1f} ms  {n:6d}x  {k[:72]}", flush=True)

# warm shapes
run_prefill(tokenizer.encode("Warmup.\n\n" + filler(60, 99, True)))

for tag, varied, seed in (("templated", False, 21), ("varied", True, 22)):
    reset()
    ids = tokenizer.encode("Repo slice.\n\n" + filler(170, seed, varied))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        wall = run_prefill(ids)
    print(f"\n{tag}: {ids.shape[-1]} tok wall {wall:.2f}s", flush=True)
    report(tag, prof)
print("\nKERNEL PROBE DONE", flush=True)
