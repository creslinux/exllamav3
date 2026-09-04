"""Per-round split under TP4+MTP (matched rules) + GPU-0 VRAM readout.
Monkey-patches: iterate_draftmodel_mtp_gen (draft loop), model.forward (verify passes),
Generator.iterate (round total). Collects rounds 20-45 after warmup, prints means.
Also reads free VRAM on device 0 with the full stack resident (replicated-head decision).
"""
import os, time, json
os.environ["EXL3_NGRAM_STREAM"] = "1"

def main():
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler, CacheLayer_fp16
    from exllamav3.generator import Generator, Job
    import exllamav3.generator.generator as genmod

    config = Config.from_directory("/models/exl3-4.05bpw")
    model = Model.from_config(config)
    draft = Model.from_config(config, component = "mtp")
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 4, max_history = 6)
    draft_cache = Cache(draft, max_num_tokens = 4096, layer_type = CacheLayer_fp16)

    for p in draft.load_gen(use_per_device = [22, 22, 22, 22], verbose = False):
        pass
    for p in model.load_gen(device = None, use_per_device = [22, 22, 22, 22],
                            tensor_p = True, tp_output_device = 0,
                            tp_backend = "native", verbose = False):
        pass
    print("loaded", flush = True)

    # VRAM readout with full stack resident (before generator buffers)
    free_b, total_b = torch.cuda.mem_get_info(0)
    print(f"GPU0 VRAM: free {free_b/1024**3:.2f} GiB / total {total_b/1024**3:.2f} GiB "
          f"(replicated head needs ~0.48 GiB)", flush = True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer,
                          draft_model = draft, draft_cache = draft_cache,
                          num_draft_tokens = 3, dynamic_draft_tokens = False,
                          record_draft_stats = True)

    acc = {"n": 0, "draft": [], "verify": [], "verify_sync": [], "total": [],
           "rewind_n": [], "rewind_t": [], "prefill_n": [], "prefill_t": []}
    SKIP, TAKE = 5, 25
    state = {"in_round": False, "draft_t": 0.0, "fwd_t": 0.0, "fwd_sync_t": 0.0,
             "rewind_n": 0, "rewind_t": 0.0, "prefill_n": 0, "prefill_t": 0.0}

    orig_dd = genmod.Generator.iterate_draftmodel_mtp_gen
    def dd(self, results):
        t0 = time.perf_counter()
        r = orig_dd(self, results)
        state["draft_t"] = time.perf_counter() - t0
        return r
    genmod.Generator.iterate_draftmodel_mtp_gen = dd

    orig_fwd = model.forward
    def fwd(*a, **kw):
        t0 = time.perf_counter()
        r = orig_fwd(*a, **kw)
        if state["in_round"]:
            state["fwd_t"] = state.get("fwd_t", 0.0) + (time.perf_counter() - t0)
            # trace-only sync: closes the verify bracket so the GPU tail is not
            # charged to the remainder at the next explicit sync / .item()
            torch.cuda.synchronize()
            state["fwd_sync_t"] = state.get("fwd_sync_t", 0.0) + (time.perf_counter() - t0)
        return r
    model.forward = fwd

    orig_da = model.tp_dispatch_all
    def da(func, args):
        t0 = time.perf_counter()
        r = orig_da(func, args)
        if state["in_round"]:
            state["rewind_n"] += 1
            state["rewind_t"] += time.perf_counter() - t0
        return r
    model.tp_dispatch_all = da

    orig_dp = draft.prefill
    def dp(*a, **kw):
        t0 = time.perf_counter()
        r = orig_dp(*a, **kw)
        if state["in_round"]:
            state["prefill_n"] += 1
            state["prefill_t"] += time.perf_counter() - t0
        return r
    draft.prefill = dp

    orig_it = genmod.Generator.iterate
    def it(self):
        state["in_round"] = True
        state["fwd_t"] = 0.0; state["fwd_sync_t"] = 0.0
        state["rewind_n"] = 0; state["rewind_t"] = 0.0
        state["prefill_n"] = 0; state["prefill_t"] = 0.0
        t0 = time.perf_counter()
        r = orig_it(self)
        dt = time.perf_counter() - t0
        state["in_round"] = False
        if r and any(not x.get("eos") for x in r) is not None:
            pass
        # count only rounds that actually drafted (draft_t set) and post-warmup
        if state["draft_t"] > 0 and acc["n"] >= 0:
            acc["n"] += 1
            if SKIP < acc["n"] <= SKIP + TAKE:
                acc["draft"].append(state["draft_t"])
                acc["verify"].append(state["fwd_t"])
                acc["verify_sync"].append(state["fwd_sync_t"])
                acc["total"].append(dt)
                acc["rewind_n"].append(state["rewind_n"])
                acc["rewind_t"].append(state["rewind_t"])
                acc["prefill_n"].append(state["prefill_n"])
                acc["prefill_t"].append(state["prefill_t"])
        return r
    genmod.Generator.iterate = it

    sampler = ComboSampler(temperature = 0.6, top_k = 20, top_p = 0.95)
    req = json.load(open("/host/healthy_long.json"))
    ids = tokenizer.encode(tokenizer.hf_render_chat_template(req["messages"]))

    # Warm job: covers the dynamic-window shape set (q_len 2..7 x history)
    job = Job(input_ids = ids.clone(), max_new_tokens = 300, sampler = sampler, identifier = "warm")
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        generator.iterate()
    print("warm job done (shapes covered)", flush = True)

    # Traced job: every verify shape has been seen; reset the accumulators
    acc["n"] = 0
    for k in ("draft", "verify", "verify_sync", "total", "rewind_n", "rewind_t", "prefill_n", "prefill_t"):
        acc[k].clear()
    job = Job(input_ids = ids.clone(), max_new_tokens = 300, sampler = sampler, identifier = "rtrace")
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        generator.iterate()

    if acc["draft"][SKIP - SKIP:][:1] or acc["draft"]:
        n = len(acc["draft"])
        d = sum(acc["draft"]) / n * 1000
        v = sum(acc["verify"]) / n * 1000
        vs = sum(acc["verify_sync"]) / n * 1000
        t = sum(acc["total"]) / n * 1000
        rn = sum(acc["rewind_n"]) / n
        rt = sum(acc["rewind_t"]) / n * 1000
        pn = sum(acc["prefill_n"]) / n
        pt = sum(acc["prefill_t"]) / n * 1000
        rem = t - d - vs
        print(f"ROUND SPLIT (x{n}): total {t:.2f}ms | draft {d:.2f} | "
              f"verify(issue) {v:.2f} verify(synced) {vs:.2f} [tail {vs-v:.2f}] | "
              f"remainder {rem:.2f}", flush = True)
        print(f"  remainder sub-buckets: rewind dispatch_all x{rn:.2f}/round = {rt:.2f}ms | "
              f"draft realign prefill x{pn:.2f}/round = {pt:.2f}ms | "
              f"other (accept/emit/sample/pagetable) {rem - rt - pt:.2f}ms", flush = True)
    stats = job.draft_stats
    tw = sum(s[1] for s in stats) or 1
    ta = sum(s[2] for s in stats)
    print(f"accept {100.0*ta/tw:.1f}%  rounds {len(stats)}  "
          f"completion[:50]={job.full_completion[:50]!r}", flush = True)
    print("ROUND TRACE DONE", flush = True)

if __name__ == "__main__":
    main()
