"""Decode-latency bench for the restored comms paths.
Run under 4 configs (env at launch):
  A: nccl backend, default           -> dist.broadcast + P2P send/recv gather + dist.all_reduce
  B: nccl, EXL3_TP_NCCL_P2P=0        -> native fallback bcast/gather + dist.all_reduce
  C: native backend, default         -> device-ring all_reduce + native bcast/gather
  D: native, EXL3_TP_DEVICE_REDUCE=0 -> CPU-path all_reduce + native bcast/gather
Metric: ms per decode step on the warmed (second) job.
"""
import os, time
os.environ.setdefault("EXL3_NGRAM_STREAM", "1")
BACKEND = os.environ.get("BENCH_TP_BACKEND", "nccl")

def main():
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, ArgmaxSampler
    from exllamav3.generator import Generator, Job

    MODEL = "/models/exl3-4.05bpw"
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 4)

    for progress in model.load_gen(
        device = None,
        use_per_device = [22, 22, 22, 22],
        tensor_p = True,
        tp_output_device = 0,
        tp_backend = BACKEND,
        verbose = False,
    ):
        pass
    print(f"loaded [{BACKEND}]", flush = True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer)
    ids = tokenizer.encode("Write a detailed essay about the history of computing, starting from mechanical calculators.")
    print(f"prompt len {ids.shape[-1]}", flush = True)

    def run_once(tag):
        job = Job(
            input_ids = ids.clone(),
            max_new_tokens = 96,
            sampler = ArgmaxSampler(),
            identifier = tag,
        )
        generator.enqueue(job)
        steps = 0
        t0 = time.perf_counter()
        while generator.num_remaining_jobs():
            generator.iterate()
            steps += 1
        dt = time.perf_counter() - t0
        ms = dt / max(steps, 1) * 1000
        print(f"[{tag}] total {dt:.2f}s  steps {steps}  {ms:.2f} ms/step  "
              f"{96 / dt:.1f} T/s  text[:40]={job.full_completion[:40]!r}", flush = True)
        return dt

    run_once("warm")
    run_once("meas")
    print("BENCH DONE", flush = True)

if __name__ == "__main__":
    main()
