"""Interleaved decode: cap 128 vs 2048 in ONE process (buffers loaded at 2048; decode rows
<=32 satisfy the guard at both caps, so toggling the module var isolates any real cap effect)."""
import os, time
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
os.environ["EXL3_TEMP_ROWS_FUSED"] = "2048"   # load buffers at 2048
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler, CacheLayer_fp16
from exllamav3.generator import Generator, Job
import exllamav3.modules.block_sparse_mlp as bsm

config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
draft = Model.from_config(config, component="mtp")
cache = Cache(model, max_num_tokens=76800, max_batch_size=8, max_history=8)
draft_cache = Cache(draft, max_num_tokens=76800, layer_type=CacheLayer_fp16)
for p in draft.load_gen(use_per_device=[22, 22, 22, 22], verbose=False): pass
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False): pass
print("loaded", flush=True)
tokenizer = Tokenizer.from_config(config)
generator = Generator(model, cache, tokenizer, draft_model=draft, draft_cache=draft_cache,
                      num_draft_tokens=3, dynamic_draft_tokens=False)
sampler = ComboSampler(temperature=0.6, top_k=20, top_p=0.95)
prompts = [f"Write {i} short paragraphs about topic {i}: distributed systems design." for i in range(1, 9)]

def conc_run():
    jobs = [Job(input_ids=tokenizer.encode(p), max_new_tokens=290, sampler=sampler) for p in prompts]
    for j in jobs: generator.enqueue(j)
    t0 = time.perf_counter(); text = ""
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if isinstance(r, dict): text += r.get("text") or ""
    return time.perf_counter() - t0, tokenizer.encode(text).shape[-1]

conc_run()  # warm
for run in range(4):
    cap = 128 if run % 2 == 0 else 2048
    bsm.TEMP_ROWS_FUSED = cap   # toggle the guard threshold only
    dt, toks = conc_run()
    print(f"INTERLEAVE cap{cap} run{run}: {toks} tokens in {dt:.2f}s = {toks/dt:.1f} T/s", flush=True)
print("INTERLEAVE DONE", flush=True)
