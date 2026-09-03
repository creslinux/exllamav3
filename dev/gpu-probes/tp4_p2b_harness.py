"""p2b fused-MoE port: parity vs the per-expert reference and timing vs bc.run_bszN.
Layer-split load (the kernel is per-shard); a GDN block's BlockSparseMLP supplies real
weights, pointer tables and the production BC path for comparison.
"""
import os, time
os.environ["EXL3_NGRAM_STREAM"] = "1"

def main():
    import torch
    import exllamav3_ext as ext
    from exllamav3 import Config, Model, Cache

    config = Config.from_directory("/models/exl3-4.05bpw")
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1)
    for p in model.load_gen(device = None, use_per_device = [22, 22, 22, 22], verbose = False):
        pass
    print("loaded", flush = True)

    # A GDN block's MoE (real weights, loaded, BC bound)
    mlp = None
    for m in model.modules:
        if type(m).__name__ == "TransformerBlock" and type(getattr(m, "attn", None)).__name__ == "GatedDeltaNet":
            mlp = m.mlp
            break
    assert mlp is not None and mlp.bc is not None, "no GDN mlp with BC found"
    dev = mlp.device
    H, E, topk = mlp.hidden_size, mlp.num_experts, mlp.num_experts_per_tok
    I = mlp.intermediate_size
    K = mlp.multi_up.K
    mcg = mlp.multi_up.mcg and mlp.multi_gate.mcg and mlp.multi_down.mcg
    print(f"mlp: dev {dev} H {H} I {I} E {E} topk {topk} K {K} mcg {mcg} "
          f"gated {mlp.gated} act {mlp.activation_fn}", flush = True)

    mg, mu, md = mlp.multi_gate, mlp.multi_up, mlp.multi_down

    def p2b(y, ids_g, w):
        bsz = y.shape[0]
        local = ids_g.reshape(-1)
        rows = torch.arange(bsz, device = dev).repeat_interleave(topk).to(torch.int32)
        ids32 = local.to(torch.int32).contiguous()
        rw = w.reshape(-1).half().contiguous()
        out = torch.empty((bsz, H), dtype = torch.half, device = dev)
        ext.p2b_fused_moe(y, out,
                          mg.ptrs_trellis, mg.ptrs_suh, mg.ptrs_svh,
                          mu.ptrs_trellis, mu.ptrs_suh, mu.ptrs_svh,
                          md.ptrs_trellis, md.ptrs_suh, md.ptrs_svh,
                          ids32, rows, rw, K, K, K, bool(mcg), bool(mlp.multi_up.mul1), H, I)
        return out

    # ---- Parity vs per-expert reference ----
    torch.manual_seed(1234)
    bsz = 2
    y = (torch.randn(bsz, H, device = dev) * 0.5).half()
    logits = torch.randn(bsz, E, device = dev)
    w, ids_g = torch.topk(logits, topk, dim = -1)
    w = torch.softmax(w.float(), dim = -1).half()

    ref = torch.zeros(bsz, H, dtype = torch.float, device = dev)
    for r in range(bsz):
        for k in range(topk):
            e = int(ids_g[r, k]); wk = w[r, k].float()
            u = mlp.ups[e].forward(y[r:r+1], {}).float()
            g = mlp.gates[e].forward(y[r:r+1], {}).float()
            a = (g / (1.0 + torch.exp(-g))) * u
            d = mlp.downs[e].forward(a.half(), {}).float()
            ref[r] += wk * d[0]

    out = p2b(y, ids_g, w).float()
    diff = (out - ref).abs()
    rel = diff.max() / ref.abs().max()
    print(f"PARITY bsz={bsz}: maxdiff {diff.max().item():.4e} refmax {ref.abs().max().item():.3f} "
          f"rel {rel.item():.3e}", flush = True)

    # ---- Timing: p2b vs bc.run_bszN ----
    def bench(fn, iters = 200, warm = 30):
        for _ in range(warm):
            fn()
        s = torch.cuda.Event(enable_timing = True); e = torch.cuda.Event(enable_timing = True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    for bsz in (1, 2, 4, 8):
        y = (torch.randn(bsz, H, device = dev) * 0.5).half()
        logits = torch.randn(bsz, E, device = dev)
        w, ids_g = torch.topk(logits, topk, dim = -1)
        w = torch.softmax(w.float(), dim = -1).half()

        t_p2b = bench(lambda: p2b(y, ids_g, w))
        t_bc = bench(lambda: mlp.bc.run_bszN(y, ids_g, w))
        print(f"bsz {bsz}: p2b {t_p2b*1000:7.1f} us | bc {t_bc*1000:7.1f} us | "
              f"ratio {t_bc/t_p2b:.2f}x", flush = True)

    print("P2B HARNESS DONE", flush = True)

if __name__ == "__main__":
    main()
