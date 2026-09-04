
"""Gate 3: long-context decode under TP4 (replicated QSA indexer x4).
Build ~100k context by repeating filler, then time 200 decode tokens."""
import os, time
os.environ["EXL3_NGRAM_STREAM"] = "1"
os.environ["EXL3_P2B_MOE"] = "1"

def main():
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler
    from exllamav3.generator import Generator, Job

    config = Config.from_directory("/models/exl3-4.05bpw")
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 131072, max_batch_size = 1)
    for p in model.load_gen(device = None, use_per_device = [22, 22, 22, 22],
                            tensor_p = True, tp_output_device = 0,
                            tp_backend = "native", verbose = False):
        pass
    print("loaded", flush = True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer)
    sampler = ComboSampler(temperature = 0.6, top_k = 20, top_p = 0.95)

    filler = ("The quick brown fox jumps over the lazy dog. " * 20000)
    ids = tokenizer.encode(filler)[..., :118000]
    n = ids.shape[-1]
    print(f"context {n} tokens", flush = True)

    t0 = time.perf_counter()
    job = Job(input_ids = ids, max_new_tokens = 1, identifier = "deep")
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        generator.iterate()
    tp = time.perf_counter() - t0
    print(f"prefill@depth: {tp:.1f}s = {n/tp:.0f} T/s", flush = True)

    # pure decode timing at depth (no draft: isolate the forward + indexer cost)
    for run in ("warm", "meas"):
        job = Job(input_ids = None, max_new_tokens = 200, sampler = sampler,
                  identifier = f"dec-{run}") if False else None
        # use a fresh job continuing the context via input of last tokens is complex;
        # simplest: one job with 201 new tokens, time from 2nd token on
        job = Job(input_ids = ids, max_new_tokens = 201, sampler = sampler, identifier = f"dec-{run}")
        generator.enqueue(job)
        t0 = time.perf_counter()
        first = None
        while generator.num_remaining_jobs():
            generator.iterate()
            if first is None and job.new_tokens >= 1:
                first = time.perf_counter()
        dt = time.perf_counter() - (first or t0)
        print(f"[{run}] decode 200 tok at depth: {dt:.2f}s = {200/dt:.1f} T/s", flush = True)

    print("LONGCTX GATE DONE", flush = True)

if __name__ == "__main__":
    main()
