"""Reasoning-controlled re-baseline: force the empty-think branch on every sample so the
branch coin-flip cannot confound either arm. Both backends, 3 measured runs, spread shown."""
import os, time, json
os.environ["EXL3_NGRAM_STREAM"] = "1"
TP = os.environ.get("BENCH_TP", "1") == "1"
# p2b is TP-only: the flag crashes at load under layer-split (known, flag-scoped)
os.environ["EXL3_P2B_MOE"] = "1" if TP else "0"

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
                            tensor_p = TP, tp_output_device = 0,
                            tp_backend = "native", verbose = False):
        pass
    tag = "TP4" if TP else "LS"
    print(f"loaded [{tag}]", flush = True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer,
                          draft_model = draft, draft_cache = draft_cache,
                          num_draft_tokens = 8, dynamic_draft_tokens = True,
                          draft_confidence = 0.6, record_draft_stats = True)
    sampler = ComboSampler(temperature = 0.6, top_k = 20, top_p = 0.95)

    def run(ids, n):
        job = Job(input_ids = ids.clone(), max_new_tokens = n, sampler = sampler,
                  identifier = "rb")
        generator.enqueue(job)
        t0 = time.perf_counter()
        while generator.num_remaining_jobs():
            generator.iterate()
        dt = time.perf_counter() - t0
        stats = job.draft_stats
        tw = sum(s[1] for s in stats) or 1
        ta = sum(s[2] for s in stats)
        emitted = sum(s[2] + 1 for s in stats)
        return dt, emitted, tw / max(len(stats), 1), 100.0 * ta / tw, job

    for name in ["short", "small1", "healthy_long", "codeheavy"]:
        req = json.load(open(f"/host/{name}.json"))
        rendered = tokenizer.hf_render_chat_template(req["messages"])
        # The template ends with the assistant opener + <think>; close it immediately so
        # every sample starts in prose -- the reasoning branch is deterministic (absent)
        # on both arms and the coin-flip variance (~40% on one prompt) is removed
        forced = rendered + "\n</think>\n\n"
        ids = tokenizer.encode(forced)
        run(ids, 300)  # warm
        rows = [run(ids, 300) for _ in range(3)]
        tps = [e / d for d, e, _, _, _ in rows]
        mean = sum(tps) / 3
        spread = max(tps) - min(tps)
        accs = [a for _, _, _, a, _ in rows]
        wins = [w for _, _, w, _, _ in rows]
        txt = rows[-1][4].full_completion[:40]
        print(f"[{tag}] {name:13s} T/s {[f'{t:.1f}' for t in tps]} mean {mean:.1f} "
              f"spread {spread:.1f} | win {[f'{w:.1f}' for w in wins]} "
              f"accept {[f'{a:.0f}' for a in accs]} | {txt!r}", flush = True)
    print("REBASE DONE", flush = True)

if __name__ == "__main__":
    main()
