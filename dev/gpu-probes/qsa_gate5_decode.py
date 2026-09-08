"""Gate 5: single-stream decode T/s on the captured path, fp16 vs q8, at 4k/64k/128k/246k."""
import os, sys, json, time
MODE = sys.argv[1]
if MODE == "q8":
    os.environ["EXL3_QSA_CACHE_BITS"] = "8"
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler, CacheLayer_fp16
from exllamav3.generator import Generator, Job

config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
draft = Model.from_config(config, component="mtp")
cache = Cache(model, max_num_tokens=262144, max_batch_size=4, max_history=8)
draft_cache = Cache(draft, max_num_tokens=262144, layer_type=CacheLayer_fp16)
for p in draft.load_gen(use_per_device=[22, 22, 22, 22], verbose=False): pass
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False): pass
print(f"loaded [{MODE}]", flush=True)
tokenizer = Tokenizer.from_config(config)
gen = Generator(model, cache, tokenizer, draft_model=draft, draft_cache=draft_cache,
                num_draft_tokens=8, dynamic_draft_tokens=True, draft_confidence=0.6)
sampler = ComboSampler(temperature=0.6, top_k=20, top_p=0.95)

CODE = ('class Container{i}:\n    """Entry {i}."""\n    def __init__(self, id, capacity={c}):\n        self.id = id; self.capacity = capacity; self.items = []\n    def add(self, sku, qty=1):\n        if len(self.items) + qty > self.capacity: return False\n        self.items.extend([sku] * qty); return True\n')

def run(ctx_tokens, seed):
    import random
    rng = random.Random(seed)
    nblocks = ctx_tokens // 92
    body = "\n\n".join(CODE.format(i=rng.randrange(10000), c=rng.randrange(8,4096)) for _ in range(nblocks))
    ids = tokenizer.encode(f"Repo slice {seed}.\n\n" + body + "\n\nImplement the next module:")
    job = Job(input_ids=ids, max_new_tokens=200, sampler=sampler)
    gen.enqueue(job)
    # prefill
    t0 = time.perf_counter()
    while gen.num_remaining_jobs() and not job.is_prefill_done():
        for r in gen.iterate(): pass
    t_pf = time.perf_counter() - t0
    # decode
    t1 = time.perf_counter(); text = ""
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if isinstance(r, dict): text += r.get("text") or ""
    t_dec = time.perf_counter() - t1
    toks = tokenizer.encode(text).shape[-1]
    return ids.shape[-1], t_pf, toks, t_dec

# warm shape (4k) discarded
run(4000, 1)
out = {}
for ctx in (4000, 64000, 128000, 246000):
    ptok, tpf, toks, tdec = run(ctx, 1000 + ctx)
    out[str(ctx)] = {"prompt_tokens": ptok, "prefill_s": round(tpf,1),
                     "gen_tokens": toks, "decode_s": round(tdec,2),
                     "decode_ts": round(toks/tdec,1)}
    print(f"{ctx}: prompt {ptok} tok, prefill {tpf:.1f}s, decode {toks} tok in {tdec:.2f}s = {toks/tdec:.1f} T/s", flush=True)
json.dump(out, open(f"/out/gate5_{MODE}.json", "w"))
print(f"GATE5 {MODE} DONE", flush=True)
