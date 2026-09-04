
"""cProfile the batch round: name the ~15ms of unattributed remainder."""
import os, time, json, cProfile, pstats
os.environ["EXL3_NGRAM_STREAM"] = "1"
os.environ["EXL3_P2B_MOE"] = "1"

def main():
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler, CacheLayer_fp16
    from exllamav3.generator import Generator, Job

    config = Config.from_directory("/models/exl3-4.05bpw")
    model = Model.from_config(config)
    draft = Model.from_config(config, component = "mtp")
    cache = Cache(model, max_num_tokens = 8192, max_batch_size = 8, max_history = 8)
    draft_cache = Cache(draft, max_num_tokens = 8192, layer_type = CacheLayer_fp16)
    for p in draft.load_gen(use_per_device = [22, 22, 22, 22], verbose = False):
        pass
    for p in model.load_gen(device = None, use_per_device = [22, 22, 22, 22],
                            tensor_p = True, tp_output_device = 0,
                            tp_backend = "native", verbose = False):
        pass
    print("loaded", flush = True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer,
                          draft_model = draft, draft_cache = draft_cache,
                          num_draft_tokens = 3, dynamic_draft_tokens = False,
                          record_draft_stats = True)
    sampler = ComboSampler(temperature = 0.6, top_k = 20, top_p = 0.95)

    names = ["short", "small1", "healthy_long", "codeheavy"]
    prompts = {}
    for name in names:
        req = json.load(open(f"/host/{name}.json"))
        prompts[name] = tokenizer.encode(tokenizer.hf_render_chat_template(req["messages"]))

    def run_batch(tag, prof=None):
        jobs = []
        for i in range(8):
            jobs.append(Job(input_ids = prompts[names[i % 4]].clone(),
                            max_new_tokens = 300, sampler = sampler,
                            identifier = f"{tag}-{i}"))
        t0 = time.perf_counter()
        for job in jobs:
            generator.enqueue(job)
        if prof:
            prof.enable()
        while generator.num_remaining_jobs():
            generator.iterate()
        if prof:
            prof.disable()
        dt = time.perf_counter() - t0
        print(f"[{tag}] wall {dt:.1f}s", flush = True)

    run_batch("warm")
    prof = cProfile.Profile()
    run_batch("meas", prof)
    st = pstats.Stats(prof)
    st.sort_stats("cumulative")
    print("== top cumulative (filter out GPU-forward wait):", flush = True)
    for func, (cc, nc, tt, ct, callers) in sorted(st.stats.items(), key=lambda x: -x[1][3]):
        filename, lineno, funcname = func
        base = os.path.basename(filename)
        if "mp_model_forward" in funcname or "tp_worker" in funcname or "recv" == funcname:
            continue
        if ct < 0.05: continue
        print(f"  {ct*1000:8.1f}ms cum {tt*1000:8.1f}ms self  {nc:6d}x  {base}:{lineno} {funcname}", flush = True)
    print("== top self:", flush = True)
    st.sort_stats("tottime")
    n = 0
    for func, (cc, nc, tt, ct, callers) in sorted(st.stats.items(), key=lambda x: -x[1][2]):
        filename, lineno, funcname = func
        base = os.path.basename(filename)
        if "mp_model_forward" in funcname or "tp_worker" in funcname or "recv" == funcname or "wait" in funcname or "spin" in funcname:
            continue
        if tt < 0.02: continue
        print(f"  {tt*1000:8.1f}ms self {nc:6d}x  {base}:{lineno} {funcname}", flush = True)
        n += 1
        if n > 14: break
    print("CPROF DONE", flush = True)

if __name__ == "__main__":
    main()
