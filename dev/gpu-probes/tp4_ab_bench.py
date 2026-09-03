"""Clean A/B: one script, both backends, matched everything.
BENCH_TP=1 -> trunk TP4 (native backend); BENCH_TP=0 -> layer split.
Matched to serving rules: ComboSampler temp 0.6 / top-k 20 / top-p 0.95,
dynamic draft (confidence 0.4), draft 6 ceiling, 1 warm + 2 measured runs.
"""
import os, time, json
os.environ["EXL3_NGRAM_STREAM"] = "1"

BATTERY = ["short", "small1", "healthy_long", "codeheavy"]
NUM_DRAFT = 6
TENSOR_P = os.environ.get("BENCH_TP", "1") == "1"
MODEL = "/models/exl3-4.05bpw"
TAG = "TP4" if TENSOR_P else "LS"

def main():
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler, CacheLayer_fp16
    from exllamav3.generator import Generator, Job

    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    draft = Model.from_config(config, component = "mtp")
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 4, max_history = NUM_DRAFT)
    draft_cache = Cache(draft, max_num_tokens = 4096, layer_type = CacheLayer_fp16)

    for p in draft.load_gen(use_per_device = [22, 22, 22, 22], verbose = False):
        pass
    for p in model.load_gen(
        device = None,
        use_per_device = [22, 22, 22, 22],
        tensor_p = TENSOR_P,
        tp_output_device = 0,
        tp_backend = "native",
        verbose = False,
    ):
        pass
    print(f"[{TAG}] loaded", flush = True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(
        model, cache, tokenizer,
        draft_model = draft,
        draft_cache = draft_cache,
        num_draft_tokens = NUM_DRAFT,
        dynamic_draft_tokens = True,
        draft_confidence = 0.4,
        record_draft_stats = True,
    )
    sampler = ComboSampler(temperature = 0.6, top_k = 20, top_p = 0.95)

    for name in BATTERY:
        req = json.load(open(f"/host/{name}.json"))
        prompt = tokenizer.hf_render_chat_template(req["messages"])
        ids = tokenizer.encode(prompt)
        runs = []
        for run in ("warm", "m1", "m2"):
            job = Job(
                input_ids = ids.clone(),
                max_new_tokens = req["max_tokens"],
                sampler = sampler,
                identifier = f"{name}-{run}",
            )
            generator.enqueue(job)
            t0 = time.perf_counter()
            results = []
            while generator.num_remaining_jobs():
                results += generator.iterate()
            dt = time.perf_counter() - t0
            stats = job.draft_stats
            if stats:
                tot_win = sum(s[1] for s in stats)
                tot_acc = sum(s[2] for s in stats)
                emitted = sum(s[2] + 1 for s in stats)
                runs.append((dt, emitted, len(stats), tot_win, tot_acc))
            else:
                errs = [r.get("error") for r in results if r.get("stage") == "error"]
                print(f"[{TAG}][{name:13s}] {run} NO STATS errors={[repr(e)[:60] for e in errs]}", flush = True)
                runs.append((dt, req["max_tokens"], 0, 0, 0))
        # report the two measured runs
        for i, (dt, emitted, rounds, tot_win, tot_acc) in enumerate(runs[1:], 1):
            print(f"[{TAG}][{name:13s}] m{i}  {dt:6.2f}s  tok {emitted:5d}  "
                  f"{emitted/dt:6.1f} T/s  rounds {rounds:4d}  "
                  f"win/pass {tot_win/max(rounds,1):4.2f}  "
                  f"accept {100.0*tot_acc/max(tot_win,1):5.1f}%", flush = True)
        (dt1, e1, r1, w1, a1), (dt2, e2, r2, w2, a2) = runs[1], runs[2]
        print(f"[{TAG}][{name:13s}] MEAN     {((e1+e2)/(dt1+dt2)):6.1f} T/s  "
              f"win/pass {(w1+w2)/max(r1+r2,1):4.2f}  "
              f"accept {100.0*(a1+a2)/max(w1+w2,1):5.1f}%", flush = True)

    print(f"[{TAG}] AB BENCH DONE", flush = True)

if __name__ == "__main__":
    main()
