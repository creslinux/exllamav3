"""Mixed-contribution parity: the decisive arm. Every 4th collective, ranks 0 and 3
contribute False (the 0-head split). Non-contributors take the CPU path -> needs a
pump thread on the master. Prediction: contributing ranks diverge (stale slot data).

Caveat for the record: the original divergence landed on rank 3 (a non-contributor)
at collective 0, with pump timeouts in the log -- the exact micro-path through the
harness's thread-based pump was never derived, and the fix removes the CPU path from
the arm entirely, so this harness alone cannot exclude harness-pump contributions.
The load-bearing evidence for the fix is the full-model validation (clean text in the
previously-corrupt p2b+one-shot cell), not this arm."""
import os, time, threading
import torch
import torch.multiprocessing as mp

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def run(rank, world):
    import exllamav3.model.model_tp_backend as tb
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    backend = tb.TPBackendNative(
        device=rank, active_devices=list(range(world)), output_device=0,
        init_method="env://", master=(rank == 0), uuid="os-mixed",
        shbuf_size=tb.SHBUF_SIZE,
    )
    if rank == 0: log("backends constructed")
    torch.cuda.synchronize()
    time.sleep(1.0)

    # pump thread on the master (what the -1 worker does in the model)
    if rank == 0:
        t = threading.Thread(target=backend.run_cpu_reduce_jobs, daemon=True)
        t.start()
        time.sleep(0.5)

    backend.oneshot_max = 64 * 1024
    n = 2560
    chain = 24
    mixed_every = 4
    inputs, refs, contribs = [], [], []
    for c in range(chain):
        torch.manual_seed(9000 + c)
        base = torch.randn(n) * 0.5
        mixed = (c % mixed_every == 0)
        contrib = False if (mixed and rank in (0, 3)) else True
        x = ((0.5 + rank * 0.25) * base).to(torch.float16).to(dev)
        if not contrib:
            x = torch.zeros_like(x)   # every False site stages zeros
        inputs.append(x)
        contribs.append(contrib)
        total = torch.zeros(n)
        for r in range(world):
            if mixed and r in (0, 3):
                continue   # non-contributors add nothing
            torch.manual_seed(9000 + c)
            total += (0.5 + r * 0.25) * (torch.randn(n) * 0.5)
        refs.append(total)

    first_bad = -1
    worst = 0.0
    for c in range(chain):
        t_ = inputs[c].clone()
        backend.all_reduce(t_, contribs[c])
        torch.cuda.synchronize()
        time.sleep(0.05)
        d = (t_.float().cpu() - refs[c]).abs().max().item()
        worst = max(worst, d)
        if d > 5e-2 and first_bad < 0:
            first_bad = c
            log(f"rank {rank}: DIVERGED at collective {c} "
                f"({'mixed' if c % mixed_every == 0 else 'all-contrib'}) maxdiff {d:.3e}")
    log(f"rank {rank}: first_bad {first_bad} worst {worst:.3e}")

    # NOTE: end the pump BEFORE the verification prints, not after -- otherwise the pump
    # thread's 50s no-jobs deadline fires during the CPU-side compares and prints a spurious
    # "CPU reduce wait timeout" that looks like a failure. Earlier revisions carried this
    # noise into committed logs.
    backend.oneshot_max = 0
    if rank == 0:
        backend.end_cpu_reduce_jobs()
        time.sleep(1.0)
    backend.close()

if __name__ == "__main__":
    mp.spawn(run, args=(4,), nprocs=4, join=True)
    log("MIXED PARITY DONE")
