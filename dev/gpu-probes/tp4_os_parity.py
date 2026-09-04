"""Chained one-shot parity vs gloo ground truth, 4 ranks, 120 chained reduces per dtype.
Chain crosses the R-buffer stage-ring wrap (~32 stages) where a slot-reuse race would fire."""
import os, multiprocessing as mp

def worker(rank, world, port, return_dict):
    import torch
    import torch.distributed as dist
    from exllamav3.model.model_tp_backend import TPBackendNative

    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world)
    dev = [0, 1, 2, 3]
    backend = TPBackendNative(
        device=rank, active_devices=dev, output_device=0,
        init_method=f"tcp://127.0.0.1:{port+1}", master=rank == dev[0],
        uuid=f"parity_{port}")
    backend.fwd_barrier()

    N = 2560
    backend.oneshot_max = 64 * 1024
    results = {}
    for dtype, tname in ((torch.float16, "fp16"), (torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        backend.fwd_barrier()
        first_bad = -1
        worst = 0.0
        for c in range(120):
            g = torch.Generator(device="cpu").manual_seed(9000 + c)
            base = torch.randn(1, N, generator=g)
            t = (base * (0.5 + rank * 0.25)).to(dtype).cuda()   # rank-dependent values
            ref = t.clone()
            dist.all_reduce(ref)                                  # gloo ground truth
            backend.all_reduce(t, True)                           # one-shot
            torch.cuda.synchronize()
            d = (t.float() - ref.float()).abs().max().item()
            worst = max(worst, d)
            if d > 5e-2 and first_bad < 0:
                first_bad = c
        results[tname] = (first_bad, worst)
    backend.oneshot_max = 0
    backend.fwd_barrier()
    if rank == 0:
        return_dict["results"] = results
    backend.close()
    dist.destroy_process_group()

def main():
    port = 29601
    mp.set_start_method("spawn", force=True)
    mgr = mp.Manager()
    rd = mgr.dict()
    procs = [mp.Process(target=worker, args=(r, 4, port, rd)) for r in range(4)]
    for p in procs: p.start()
    for p in procs: p.join(120)
    for p in procs:
        if p.is_alive(): p.terminate()
    if "results" in rd:
        print("PARITY (first_bad_index, worst_maxdiff):", dict(rd["results"]), flush=True)
    else:
        print("PARITY: no result (hang or crash)", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
