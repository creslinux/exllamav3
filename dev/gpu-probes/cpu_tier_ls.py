import os, time
os.environ["EXL3_P2B_MOE"] = "1"
os.environ["EXL3_P2B_QUIET"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler
from exllamav3.generator import Generator, Job

config = Config.from_directory("/models/exl3-4.05bpw")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=4096)
for p in model.load_gen(use_per_device=[16, 22, 22, 22], verbose=False):
    pass
print("loaded (pool 4096, tier 512MB)", flush=True)
tokenizer = Tokenizer.from_config(config)
generator = Generator(model, cache, tokenizer, cpu_cache_size=512 * 1024 * 1024)
tier = generator.cpu_page_cache
greedy = ComboSampler(temperature=0.0)

secret = "ZEPHYR-7419"
secret2 = "QUARTZ-3307"
fa = ("Quarterly logistics report: depot alpha inventory reconciled. ") * 128 + " Final tally code: QUARTZ-3307."
fb = ("Orchard irrigation memo: block seven valve pressure nominal. ") * 310  # ~6.5k, 2x residuum -> evicts A

def metrics(tag):
    m = dict(tier.metrics)
    print(f"{tag}: pushes={m['pushes']} restores={m['restores']} evictions={m['evictions']}", flush=True)

def run(ids, n, tag):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=greedy)
    t0 = time.perf_counter()
    generator.enqueue(job)
    text = ""
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            text += r.get("text", "")
    dt = time.perf_counter() - t0
    print(f"{tag}: {ids.shape[-1]} tok in, {dt:.1f}s", flush=True)
    metrics(tag)
    return text

a1_ids = tokenizer.encode(f"The codeword is {secret}. Memorize it.\n\n" + fa)
run(a1_ids, 8, "A1 (seed ~7pp)     ")
b_ids = tokenizer.encode("Different memo, ignore.\n\n" + fb)
run(b_ids, 8, "B  (evict ~3.7xpool)")
q_ids = tokenizer.encode(" Repeat back both codewords, start and final tally. They are:")
a2_ids = torch.cat([a1_ids, q_ids], dim=-1)
out = run(a2_ids, 60, "A2 (recall, prefix) ")
print(f"A2 out: {out.strip()[:80]!r}", flush=True)
print("start-codeword correct:", secret in out, flush=True)
print("tail-codeword (evicted zone) correct:", secret2 in out, flush=True)
print("TIER TEST4 DONE", flush=True)
