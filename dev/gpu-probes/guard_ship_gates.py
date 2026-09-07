"""Ship-gates for the relaxed guard + cap 2048: (1) read completions, (2) 8-stream
concurrency vs stock, (3) VRAM. Env: EXL3_TEMP_ROWS_FUSED (patched pkg) or stock."""
import os, sys, time, random
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler, CacheLayer_fp16
from exllamav3.generator import Generator, Job

CAP = os.environ.get("EXL3_TEMP_ROWS_FUSED", "128")
config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
draft = Model.from_config(config, component="mtp")
cache = Cache(model, max_num_tokens=8192, max_batch_size=8, max_history=8)
draft_cache = Cache(draft, max_num_tokens=8192, layer_type=CacheLayer_fp16)
for p in draft.load_gen(use_per_device=[22, 22, 22, 22], verbose=False):
    pass
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False):
    pass
print(f"loaded (cap {CAP})", flush=True)
tokenizer = Tokenizer.from_config(config)
generator = Generator(model, cache, tokenizer, draft_model=draft, draft_cache=draft_cache,
                      num_draft_tokens=3, dynamic_draft_tokens=False, record_draft_stats=True)
sampler = ComboSampler(temperature=0.6, top_k=20, top_p=0.95)

def run_jobs(jobs):
    t0 = time.perf_counter()
    for j in jobs: generator.enqueue(j)
    errors, text = [], ""
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if not isinstance(r, dict): continue
            if r.get("error"):
                errors.append(str(r.get("error"))[:300])
            text += r.get("text") or ""
    return time.perf_counter() - t0, errors, text

# --- gate 1: read completions (single stream, guard active at decode shapes) ---
for run in range(2):
    job = Job(input_ids=tokenizer.encode("Write a Python function that merges two sorted lists, with a docstring."),
              max_new_tokens=300, sampler=sampler)
    dt, errs, text = run_jobs([job])
    if errs: print(f"G1 run{run} ERROR: {errs[0]}", flush=True)
    print(f"G1 run{run}: {len(text)} chars in {dt:.1f}s | FULL: {text!r}", flush=True)

# --- gate 2: 8-stream concurrency, static window 3 ---
prompts = [f"Write {i} short paragraphs about topic {i}: distributed systems design." for i in range(1, 9)]
for warm in range(1):
    jobs = [Job(input_ids=tokenizer.encode(p), max_new_tokens=64, sampler=sampler) for p in prompts]
    run_jobs(jobs)
jobs = [Job(input_ids=tokenizer.encode(p), max_new_tokens=290, sampler=sampler) for p in prompts]
dt, errs, text = run_jobs(jobs)
if errs: print(f"G2 ERROR: {errs[0]}", flush=True)
import re as _re
toks = len(_re.findall(r"\S+", text))
print(f"G2 cap{CAP}: 8 streams, {dt:.1f}s wall, ~{toks} words emitted", flush=True)
print(f"GATE RIG DONE cap{CAP}", flush=True)
