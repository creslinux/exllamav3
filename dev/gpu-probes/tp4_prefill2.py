"""Prefill gate, corrected protocol: no prompt ever repeated (prefix cache must never fire).
A: throwaway ~4k (burns autotune + first table paging)
B: throwaway ~4k, different content (delta A->B = incremental disk-table paging alone)
then each battery prompt exactly once, including the 50k-depth file."""
import os, time, json
os.environ["EXL3_NGRAM_STREAM"] = "1"
TP = os.environ.get("BENCH_TP", "1") == "1"

def make_throwaway(seed: int, n_words: int) -> str:
    import random
    rng = random.Random(seed)
    words = []
    vocab = ["harbor", "matrix", "cinder", "vellum", "orchard", "talon", "quartz", "meadow",
             "fjord", "lantern", "copper", "thistle", "cascade", "pebble", "sumac", "glacier"]
    for i in range(n_words):
        words.append(f"{vocab[rng.randrange(len(vocab))]}{rng.randrange(9999)}")
    return " ".join(words)

def main():
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.generator import Generator, Job

    config = Config.from_directory("/models/exl3-4.05bpw")
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 61440, max_batch_size = 1)
    for p in model.load_gen(device = None, use_per_device = [22, 22, 22, 22],
                            tensor_p = TP, tp_output_device = 0,
                            tp_backend = "native", verbose = False):
        pass
    tag = "TP4" if TP else "LS"
    print(f"loaded [{tag}]", flush = True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer)

    def prefill_once(ids, label):
        n = ids.shape[-1]
        job = Job(input_ids = ids, max_new_tokens = 1, identifier = label)
        generator.enqueue(job)
        t0 = time.perf_counter()
        while generator.num_remaining_jobs():
            generator.iterate()
        dt = time.perf_counter() - t0
        print(f"[{tag}] {label}: {n} tok, {dt:.2f}s = {n/dt:.0f} T/s", flush = True)
        return dt

    a = prefill_once(tokenizer.encode(make_throwaway(1, 3200)), "throwA(autotune+table)")
    b = prefill_once(tokenizer.encode(make_throwaway(2, 3200)), "throwB(table-delta)")
    print(f"[{tag}] incremental disk-table paging (B-A): {b-a:.2f}s over ~4k tok", flush = True)

    for name in ("pf_4k", "pf_16k", "pf_32k", "big50k"):
        req = json.load(open(f"/host/{name}.json"))
        content = req["messages"][0]["content"] if "messages" in req else req["prompt"]
        prefill_once(tokenizer.encode(content), name)

    print("PREFILL GATE DONE", flush = True)

if __name__ == "__main__":
    main()
