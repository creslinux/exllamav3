"""Gate 6: like-for-like VRAM (draft loaded, in-process) + 512k pool fit."""
import os, sys, json
MODE, CACHE = sys.argv[1], int(sys.argv[2])
if MODE == "q8":
    os.environ["EXL3_QSA_CACHE_BITS"] = "8"
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, CacheLayer_fp16
from exllamav3.generator import Generator, Job

config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
draft = Model.from_config(config, component="mtp")
cache = Cache(model, max_num_tokens=CACHE, max_batch_size=8, max_history=8)
draft_cache = Cache(draft, max_num_tokens=CACHE, layer_type=CacheLayer_fp16)
for p in draft.load_gen(use_per_device=[22, 22, 22, 22], verbose=False): pass
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False): pass
torch.cuda.synchronize()
alloc = {str(d): torch.cuda.memory_allocated(d)//2**20 for d in range(4)}
print(f"[{MODE} cache {CACHE}] VRAM after load (draft loaded): " + " ".join(f"{d}:{alloc[str(d)]}MiB" for d in range(4)), flush=True)
json.dump({"mode": MODE, "cache": CACHE, "alloc": alloc}, open(f"/out/gate6_{MODE}_{CACHE}.json", "w"))
print(f"GATE6 {MODE} {CACHE} DONE", flush=True)
