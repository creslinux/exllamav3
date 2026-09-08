"""Gate 1: block-selection identity fp16 vs Q8 + end-of-prompt logits (first token)."""
import os, sys, json
MODE = sys.argv[1]
if MODE == "q8":
    os.environ["EXL3_QSA_CACHE_BITS"] = "8"
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
os.environ["EXL3_BC_ATTN_TRACE"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler
from exllamav3.generator import Generator, Job

config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=40192)
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False):
    pass
print(f"loaded [{MODE}]", flush=True)

import exllamav3.modules.qsa_indexer as qsimod
captured = []
_orig = qsimod.QSAIndexer.select_indices_paged
def _wrap(self, layer, q_idx, block_table, cache_seqlens_cpu):
    out = _orig(self, layer, q_idx, block_table, cache_seqlens_cpu)
    captured.append(out.clone().cpu())
    return out
qsimod.QSAIndexer.select_indices_paged = _wrap

tokenizer = Tokenizer.from_config(config)
gen = Generator(model, cache, tokenizer)
greedy = ComboSampler(temperature=0.0)
filler = ("The quarterly report covers depot logistics, regional staffing rotations, and "
          "deferred maintenance schedules for the period. ") * 950
ids = tokenizer.encode("Context follows.\n\n" + filler + "\n\nThe capital of France is")
job = Job(input_ids=ids, max_new_tokens=8, sampler=greedy, return_logits=True)
gen.enqueue(job)
text = ""
errs = []
first_logits = None
while gen.num_remaining_jobs():
    for r in gen.iterate():
        if isinstance(r, dict):
            if r.get("error"): errs.append(str(r.get("error"))[:200])
            if r.get("logits") is not None and first_logits is None:
                first_logits = r["logits"]
            text += r.get("text") or ""

torch.save(captured, f"/out/gate1_idx_{MODE}_list.pt")
print(f"captured {len(captured)} index tensors -> gate1_idx_{MODE}.pt", flush=True)

if first_logits is not None:
    lg = first_logits
    l = lg[0].reshape(-1).float()   # first held position, (V,)
    v, i = torch.topk(l, 3)
    lp = torch.log_softmax(l, -1)
    print(f"LOGITS shape {list(lg.shape)} top-1 {i[0].item()} logprob {lp[i[0]].item():.6f} top-3 {i.tolist()}", flush=True)
    json.dump({"mode": MODE, "top1": int(i[0].item()), "logprob": float(lp[i[0]].item()),
               "top3": i.tolist(), "n_captured": len(captured)},
              open(f"/out/gate1_{MODE}.json", "w"))
if errs: print(f"ERROR: {errs[0]}", flush=True)
print(f"OUTPUT: {text!r}", flush=True)
print(f"GATE1 {MODE} DONE", flush=True)
