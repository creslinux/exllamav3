"""Gate 2: end-of-prompt top-k logits at 16k/128k, prose + code. Modes fp16_r1/fp16_r2/q8.
One process per mode, four distinct prompts, first-token greedy logits captured."""
import os, sys, json, random
MODE = sys.argv[1]
if MODE == "q8":
    os.environ["EXL3_QSA_CACHE_BITS"] = "8"
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler
from exllamav3.generator import Generator, Job

config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=140288)   # fits 128k prompt + margin
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False):
    pass
print(f"loaded [{MODE}]", flush=True)
tokenizer = Tokenizer.from_config(config)
gen = Generator(model, cache, tokenizer)
greedy = ComboSampler(temperature=0.0)

SENT = "The quarterly report covers depot logistics, regional staffing rotations, and deferred maintenance schedules for the period. "
CODE = ('class Container{i}:\n    """Registry entry {i}."""\n    def __init__(self, id, capacity={c}):\n        self.id = id; self.capacity = capacity; self.items = []\n    def add(self, sku, qty=1):\n        if len(self.items) + qty > self.capacity: return False\n        self.items.extend([sku] * qty); return True\n')

def prompt(tag, seed):
    rng = random.Random(seed)
    if "prose" in tag:
        body = SENT * (int(tag.split("k")[0]) * 1000 // 18)
    else:
        body = "\n\n".join(CODE.format(i=rng.randrange(10000), c=rng.randrange(8,4096)) for _ in range(int(tag.split("k")[0]) * 1000 // 92))
    return f"Document {seed}.\n\n" + body + "\n\nSummarize the key findings in one sentence:"

results = {}
for tag, seed in [("16k-prose", 101), ("16k-code", 102), ("128k-prose", 103), ("128k-code", 104)]:
    ids = tokenizer.encode(prompt(tag, seed))
    job = Job(input_ids=ids, max_new_tokens=1, sampler=greedy, return_logits=True)
    gen.enqueue(job)
    lg = None; errs = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if isinstance(r, dict):
                if r.get("error"): errs.append(str(r.get("error"))[:200])
                if r.get("logits") is not None and lg is None: lg = r["logits"]
    if errs: print(f"{tag} ERROR: {errs[0]}", flush=True)
    l = lg[0].reshape(-1).float()
    v, i = torch.topk(l, 3)
    lp = torch.log_softmax(l, -1)
    results[tag] = {"tokens": int(ids.shape[-1]), "top1": int(i[0].item()),
                    "top3": i.tolist(), "logprob": float(lp[i[0]].item())}
    print(f"{tag}: {ids.shape[-1]} tok | top-1 {i[0].item()} logprob {lp[i[0]].item():.6f} top-3 {i.tolist()}", flush=True)

json.dump(results, open(f"/out/gate2_{MODE}.json", "w"))
print(f"GATE2 {MODE} DONE", flush=True)
