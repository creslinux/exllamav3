"""One-shot collective parity, 4 ranks, adapted from the proven gate_bench_native pattern.
Chain 16 (below the 120-stage ring) vs chain 120 (crosses it). Per-collective divergence
index reported. Reference = seeded per-rank reconstruction, no gloo."""
import os, time
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
        init_method="env://", master=(rank == 0), uuid="os-parity",
        shbuf_size=tb.SHBUF_SIZE,
    )
    if rank == 0: log("backends constructed")
    torch.cuda.synchronize()
    time.sleep(1.0)

    backend.oneshot_max = 64 * 1024
    n = 2560
    for dtype, tname in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
      for chain in (240,):
        # Rebuild identical inputs per collective; reference = sum of all ranks' contributions
        inputs, refs = [], []
        for c in range(chain):
            torch.manual_seed(9000 + c)
            base = torch.randn(n) * 0.5
            inputs.append(((0.5 + rank * 0.25) * base).to(dtype).to(dev))
            total = torch.zeros(n)
            for r in range(world):
                torch.manual_seed(9000 + c)
                total += (0.5 + r * 0.25) * (torch.randn(n) * 0.5)
            refs.append(total)
        # Phase 1: run the whole chain at model pace (no intermediate sync)
        outs = []
        for c in range(chain):
            t = inputs[c].clone()
            backend.all_reduce(t, True)
            outs.append(t)
        torch.cuda.synchronize()
        time.sleep(0.5)
        # Phase 2: verify all
        first_bad = -1
        worst = 0.0
        for c in range(chain):
            d = (outs[c].float().cpu() - refs[c]).abs().max().item()
            worst = max(worst, d)
            if d > 5e-2 and first_bad < 0:
                first_bad = c
                if rank == 0:
                    log(f"{tname} chain {chain}: DIVERGED at collective {c} maxdiff {d:.3e}")
        if rank == 0:
            log(f"{tname} chain {chain}: first_bad {first_bad} worst {worst:.3e}")
    backend.oneshot_max = 0
    time.sleep(0.5)
    backend.close()

if __name__ == "__main__":
    mp.spawn(run, args=(4,), nprocs=4, join=True)
    log("OS PARITY DONE")
