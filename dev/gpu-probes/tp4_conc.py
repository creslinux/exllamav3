"""Gate 1: concurrency under TP4. Eight concurrent streams through the generator,
aggregate and per-stream throughput. The p2b engagement counter will show whether the
fast path survives batch decode or the 8x4=32-row verify falls off it."""
import os, time, json
os.environ["EXL3_NGRAM_STREAM"] = "1"
os.environ["EXL3_P2B_MOE"] = "1"

def main():
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler, CacheLayer_fp16
    from exllamav3.generator import Generator, Job

    config = Config.from_directory("/models/exl3-4.05bpw")
    model = Model.from_config(config)
    draft = Model.from_config(config, component = "mtp")
    N_STREAMS = int(os.environ.get("N_STREAMS", "8"))
    cache = Cache(model, max_num_tokens = 8192, max_batch_size = N_STREAMS, max_history = 8)
    draft_cache = Cache(draft, max_num_tokens = 8192, layer_type = CacheLayer_fp16)
    for p in draft.load_gen(use_per_device = [22, 22, 22, 22], verbose = False):
        pass
    for p in model.load_gen(device = None, use_per_device = [22, 22, 22, 22],
                            tensor_p = True, tp_output_device = 0,
                            tp_backend = "native", verbose = False):
        pass
    print(f"loaded TP4, {N_STREAMS} streams", flush = True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer,
                          draft_model = draft, draft_cache = draft_cache,
                          num_draft_tokens = 8, dynamic_draft_tokens = True,
                          draft_confidence = 0.6, record_draft_stats = True)
    sampler = ComboSampler(temperature = 0.6, top_k = 20, top_p = 0.95)

    names = ["short", "small1", "healthy_long", "codeheavy"]
    prompts = {}
    for name in names:
        req = json.load(open(f"/host/{name}.json"))
        prompts[name] = tokenizer.encode(tokenizer.hf_render_chat_template(req["messages"]))

    # warm pass (shapes, calibrator) then measured pass, each with N_STREAMS concurrent jobs
    def run_batch(tag):
        jobs = []
        for i in range(N_STREAMS):
            name = names[i % len(names)]
            jobs.append((name, Job(input_ids = prompts[name].clone(),
                                   max_new_tokens = 300, sampler = sampler,
                                   identifier = f"{tag}-{i}")))
        t0 = time.perf_counter()
        for _, job in jobs:
            generator.enqueue(job)
        while generator.num_remaining_jobs():
            generator.iterate()
        dt = time.perf_counter() - t0
        total = sum(len(j.full_completion) for _, j in jobs)  # chars, only for ratio; use rounds below
        per = [(n, len(j.draft_stats), 0) for n, j in jobs]
        emitted = sum(sum(s[2] + 1 for s in j.draft_stats) for _, j in jobs)
        aggr = emitted / dt
        print(f"[{tag}] wall {dt:.1f}s emitted {emitted} aggregate {aggr:.1f} T/s "
              f"rounds/stream {[t for _, t, _ in per]}", flush = True)
        return aggr

    run_batch("warm")
    run_batch("meas1")
    run_batch("meas2")
    print("CONCURRENCY GATE DONE", flush = True)

if __name__ == "__main__":
    main()
