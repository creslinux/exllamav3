"""Concurrency per-round split at 8 streams: where does the batch round's time go?
Static window 3 (window-independent ms/round), warm + 2 measured, split instrumentation."""
import os, time, json
os.environ["EXL3_NGRAM_STREAM"] = "1"
import sys
TP = os.environ.get("BENCH_TP", "1") == "1"
# LS+p2b fixed on this branch (b49f8db); default keeps historical behavior, env overrides either arm
os.environ["EXL3_P2B_MOE"] = os.environ.get("EXL3_P2B_MOE", "1" if TP else "0")

def main():
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler, CacheLayer_fp16
    from exllamav3.generator import Generator, Job
    import exllamav3.generator.generator as genmod

    config = Config.from_directory("/models/exl3-4.05bpw")
    model = Model.from_config(config)
    draft = Model.from_config(config, component = "mtp")
    cache = Cache(model, max_num_tokens = 8192, max_batch_size = 8, max_history = 8)
    draft_cache = Cache(draft, max_num_tokens = 8192, layer_type = CacheLayer_fp16)
    for p in draft.load_gen(use_per_device = [22, 22, 22, 22], verbose = False):
        pass
    for p in model.load_gen(device = None, use_per_device = [22, 22, 22, 22],
                            tensor_p = TP, tp_output_device = 0,
                            tp_backend = "native", verbose = False):
        pass
    print(f"loaded [{chr(84)+chr(80)+chr(52) if TP else chr(76)+chr(83)}]", flush = True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer,
                          draft_model = draft, draft_cache = draft_cache,
                          num_draft_tokens = 3, dynamic_draft_tokens = False,
                          record_draft_stats = True)
    sampler = ComboSampler(temperature = 0.6, top_k = 20, top_p = 0.95)

    acc = {"draft": [], "verify_sync": [], "total": []}
    state = {"in_round": False, "draft_t": 0.0, "fwd_t": 0.0, "fwd_sync_t": 0.0}
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
            torch.cuda.synchronize()
            state["fwd_sync_t"] = state.get("fwd_sync_t", 0.0) + (time.perf_counter() - t0)
        return r
    model.forward = fwd
    orig_it = genmod.Generator.iterate
    def it(self):
        state["in_round"] = True
        state["fwd_t"] = 0.0; state["fwd_sync_t"] = 0.0
        t0 = time.perf_counter()
        r = orig_it(self)
        dt = time.perf_counter() - t0
        state["in_round"] = False
        acc["draft"].append(state["draft_t"])
        acc["verify_sync"].append(state["fwd_sync_t"])
        acc["total"].append(dt)
        return r
    genmod.Generator.iterate = it

    names = ["short", "small1", "healthy_long", "codeheavy"]
    prompts = {}
    for name in names:
        req = json.load(open(f"/host/{name}.json"))
        prompts[name] = tokenizer.encode(tokenizer.hf_render_chat_template(req["messages"]))

    def run_batch(tag):
        jobs = []
        for i in range(8):
            jobs.append(Job(input_ids = prompts[names[i % 4]].clone(),
                            max_new_tokens = 300, sampler = sampler,
                            identifier = f"{tag}-{i}"))
        n0 = len(acc["total"])
        t0 = time.perf_counter()
        for job in jobs:
            generator.enqueue(job)
        while generator.num_remaining_jobs():
            generator.iterate()
        dt = time.perf_counter() - t0
        rounds = acc["total"][n0:]
        n = len(rounds)
        d = sum(acc["draft"][n0:]) / n * 1000
        v = sum(acc["verify_sync"][n0:]) / n * 1000
        t = sum(rounds) / n * 1000
        emitted = 300 * 8
        print(f"[{tag}] wall {dt:.1f}s aggregate {emitted/dt:.1f} T/s | "
              f"round(total) {t:.2f}ms draft {d:.2f} verify(synced) {v:.2f} "
              f"remainder {t-d-v:.2f} | rounds {n}", flush = True)
        return emitted / dt

    run_batch("warm")
    run_batch("meas1")
    run_batch("meas2")
    print("CONC SPLIT DONE", flush = True)

if __name__ == "__main__":
    main()
