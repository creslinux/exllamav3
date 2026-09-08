"""Gate 4: full-context (~260k) request under Q8, 0 errors, per-card VRAM before/after."""
import os
os.environ["EXL3_QSA_CACHE_BITS"] = "8"
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.generator import Generator, Job

config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=262144)
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False):
    pass
print("loaded [q8, pool 262144]", flush=True)
def vram(tag):
    torch.cuda.synchronize()
    print(f"VRAM {tag}: " + " ".join(f"{d}:{torch.cuda.memory_allocated(d)//2**20}MiB" for d in range(4)), flush=True)
vram("after-load")
tokenizer = Tokenizer.from_config(config)
gen = Generator(model, cache, tokenizer)
sent = "The quarterly report covers depot logistics, regional staffing rotations, and deferred maintenance schedules for the period. "
ids = tokenizer.encode("Context follows.\n\n" + sent * 13700)
print(f"prompt {ids.shape[-1]} tokens", flush=True)
job = Job(input_ids=ids, max_new_tokens=1)
gen.enqueue(job)
errs = []
while gen.num_remaining_jobs():
    for r in gen.iterate():
        if isinstance(r, dict) and r.get("error"): errs.append(str(r.get("error"))[:200])
vram("after-prefill")
if errs: print(f"ERROR: {errs[0]}", flush=True)
else: print("0 errors", flush=True)
print("GATE4 DONE", flush=True)
