"""Verify decomposition (run 1) and per-row scaling (run 2) via the TRACE_STEP probe.
Run 1 (MODE=decomp): dynamic draft (matched rules), healthy_long + codeheavy.
Run 2 (MODE=scale): static draft window DRAFT_N in {1,3,5,7} -> verify q_len in {2,4,6,8}.
Long jobs: the probe re-arms every 50 passes; the LAST summary per job is the measurement
(all shapes seen, 100+ warm passes before it).
"""
import os, time, json
os.environ["EXL3_NGRAM_STREAM"] = "1"
os.environ["EXL3_TP_TRACE_STEP"] = "1"

MODE = os.environ.get("MODE", "decomp")
DRAFT_N = int(os.environ.get("DRAFT_N", "5"))

def main():
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler, CacheLayer_fp16
    from exllamav3.generator import Generator, Job

    config = Config.from_directory("/models/exl3-4.05bpw")
    model = Model.from_config(config)
    draft = Model.from_config(config, component = "mtp")
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 4, max_history = 8)
    draft_cache = Cache(draft, max_num_tokens = 4096, layer_type = CacheLayer_fp16)

    for p in draft.load_gen(use_per_device = [22, 22, 22, 22], verbose = False):
        pass
    for p in model.load_gen(device = None, use_per_device = [22, 22, 22, 22],
                            tensor_p = True, tp_output_device = 0,
                            tp_backend = "native", verbose = False):
        pass
    print("loaded", flush = True)

    tokenizer = Tokenizer.from_config(config)
    if MODE == "decomp":
        generator = Generator(model, cache, tokenizer,
                              draft_model = draft, draft_cache = draft_cache,
                              num_draft_tokens = 6, dynamic_draft_tokens = True,
                              draft_confidence = 0.4, record_draft_stats = True)
        prompts = ["healthy_long", "codeheavy"]
    else:
        generator = Generator(model, cache, tokenizer,
                              draft_model = draft, draft_cache = draft_cache,
                              num_draft_tokens = DRAFT_N, dynamic_draft_tokens = False,
                              record_draft_stats = True)
        prompts = ["healthy_long"]

    sampler = ComboSampler(temperature = 0.6, top_k = 20, top_p = 0.95)
    for name in prompts:
        req = json.load(open(f"/host/{name}.json"))
        ids = tokenizer.encode(tokenizer.hf_render_chat_template(req["messages"]))
        print(f"=== JOB START {name} (mode={MODE} draft_n={DRAFT_N if MODE=='scale' else 'dyn'})",
              flush = True)
        job = Job(input_ids = ids, max_new_tokens = 500, sampler = sampler,
                  identifier = f"{name}")
        generator.enqueue(job)
        while generator.num_remaining_jobs():
            generator.iterate()
        stats = job.draft_stats
        tw = sum(s[1] for s in stats) or 1
        ta = sum(s[2] for s in stats)
        print(f"=== JOB END {name}: accept {100.0*ta/tw:.1f}% rounds {len(stats)} "
              f"win/pass {tw/len(stats):.2f}", flush = True)

    print("PROBE RUN DONE", flush = True)

if __name__ == "__main__":
    main()
