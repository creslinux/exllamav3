"""Gate 3: long-generation drift. 2k tokens each on prose + code, MTP draft conf 0.6, acceptance + tail."""
import os, sys, json
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
cache = Cache(model, max_num_tokens=8192, max_batch_size=4, max_history=8)
draft_cache = Cache(draft, max_num_tokens=8192, layer_type=CacheLayer_fp16)
for p in draft.load_gen(use_per_device=[22, 22, 22, 22], verbose=False): pass
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False): pass
print(f"loaded [{MODE}]", flush=True)
tokenizer = Tokenizer.from_config(config)
gen = Generator(model, cache, tokenizer, draft_model=draft, draft_cache=draft_cache,
                num_draft_tokens=8, dynamic_draft_tokens=True, draft_confidence=0.6,
                record_draft_stats=True)
sampler = ComboSampler(temperature=0.6, top_k=20, top_p=0.95)

prompts = [
    ("prose", "Write a detailed technical essay about the trade-offs of distributed consensus algorithms, covering Raft and Paxos, their failure modes, and when to prefer one over the other."),
    ("code",  "Write a Python module implementing a thread-safe bounded queue with producer/consumer helpers, retry logic, and a small test suite."),
]
out = {}
for tag, p in prompts:
    ids = tokenizer.encode(p)
    job = Job(input_ids=ids, max_new_tokens=2048, sampler=sampler)
    gen.enqueue(job)
    text = ""; errs = []; acc = ret = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if isinstance(r, dict):
                if r.get("error"): errs.append(str(r.get("error"))[:200])
                text += r.get("text") or ""
                if r.get("accepted_draft_tokens") is not None:
                    acc = r["accepted_draft_tokens"]; ret = r["rejected_draft_tokens"]
    a = acc or 0; rj = ret or 0
    accept = a / max(1, a + rj)
    out[tag] = {"chars": len(text), "accepted": a, "rejected": rj, "acceptance": accept,
                "tail": text[-200:]}
    if errs: print(f"{tag} ERROR: {errs[0]}", flush=True)
    print(f"{tag}: {len(text)} chars | accepted {a} rejected {rj} = {accept:.2%} accept | tail: {text[-120:]!r}", flush=True)
json.dump(out, open(f"/out/gate3_{MODE}.json", "w"))
print(f"GATE3 {MODE} DONE", flush=True)
