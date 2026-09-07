"""Concurrency (token-counted, 2 measured runs) + prefill logits gate. Env: EXL3_TEMP_ROWS_FUSED.
Cap read from env: 128 = stock-guard-baseline OR guard-only (patched pkg); 2048 = guard+cap."""
import os, sys, time
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler, CacheLayer_fp16
from exllamav3.generator import Generator, Job

CAP = os.environ.get("EXL3_TEMP_ROWS_FUSED", "128")
config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
draft = Model.from_config(config, component="mtp")
cache = Cache(model, max_num_tokens=76800, max_batch_size=8, max_history=8)
draft_cache = Cache(draft, max_num_tokens=76800, layer_type=CacheLayer_fp16)
for p in draft.load_gen(use_per_device=[22, 22, 22, 22], verbose=False): pass
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False): pass
print(f"loaded (cap {CAP})", flush=True)
tokenizer = Tokenizer.from_config(config)
generator = Generator(model, cache, tokenizer, draft_model=draft, draft_cache=draft_cache,
                      num_draft_tokens=3, dynamic_draft_tokens=False)
sampler = ComboSampler(temperature=0.6, top_k=20, top_p=0.95)

def run_jobs(jobs):
    t0 = time.perf_counter()
    for j in jobs: generator.enqueue(j)
    errors, text = [], ""
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if not isinstance(r, dict): continue
            if r.get("error"): errors.append(str(r.get("error"))[:300])
            text += r.get("text") or ""
    return time.perf_counter() - t0, errors, text

def conc_run():
    prompts = [f"Write {i} short paragraphs about topic {i}: distributed systems design." for i in range(1, 9)]
    jobs = [Job(input_ids=tokenizer.encode(p), max_new_tokens=290, sampler=sampler) for p in prompts]
    dt, errs, text = run_jobs(jobs)
    toks = tokenizer.encode(text).shape[-1]
    return dt, toks, errs

# warm
conc_run()
# measured x2
for run in range(2):
    dt, toks, errs = conc_run()
    if errs: print(f"CONC cap{CAP} run{run} ERROR: {errs[0]}", flush=True)
    print(f"CONC cap{CAP} run{run}: {toks} tokens in {dt:.2f}s = {toks/dt:.1f} T/s aggregate", flush=True)

# --- prefill logits gate: 16k prompt, top-1 + logprob at end of prompt ---
filler = ("The quarterly report covers depot logistics, regional staffing rotations, and "
          "deferred maintenance schedules for the period. ") * 830  # ~16k tokens
ids = tokenizer.encode("Context document:\n\n" + filler)
print(f"logits prompt: {ids.shape[-1]} tokens", flush=True)
job = Job(input_ids=ids, max_new_tokens=1, sampler=ComboSampler(temperature=0.0), return_logits=True)
generator.enqueue(job)
errs = []
while generator.num_remaining_jobs():
    for r in generator.iterate():
        if isinstance(r, dict) and r.get("logits") is not None:
            lg = r["logits"]
            l = lg.float().reshape(-1)  # last-position logits, vocab-length
            v, i = torch.topk(l, 3)
            lp = torch.log_softmax(l, -1)
            print(f"LOGITS cap{CAP}: top-1 tok {i[0].item()} logprob {lp[i[0]].item():.6f} | "
                  f"top-3 {i.tolist()}", flush=True)
        elif isinstance(r, dict) and r.get("error"):
            errs.append(str(r.get("error"))[:300])
            print(f"LOGITS cap{CAP} ERROR: {errs[-1]}", flush=True)
print(f"DONE cap{CAP}", flush=True)
