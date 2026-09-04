"""Three-way discriminator for the p2b parity failure: one expert, one row.
(a) Linear wrapper forwards (the harness reference)
(b) manual reconstruct + had_r_128 + hgemm chain (format-level ground truth)
(c) p2b single slot, weight 1.0
"""
import os
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

    mlp = None
    for m in model.modules:
        if type(m).__name__ == "TransformerBlock" and type(getattr(m, "attn", None)).__name__ == "GatedDeltaNet":
            mlp = m.mlp
            break
    dev = mlp.device
    H, I = mlp.hidden_size, mlp.intermediate_size
    mg, mu, md = mlp.multi_gate, mlp.multi_up, mlp.multi_down
    K = mu.K
    mcg = bool(mu.mcg)
    print(f"mlp dev {dev} H {H} I {I} K {K} mcg {mcg} mul1 {bool(mlp.multi_up.mul1)}", flush = True)

    torch.manual_seed(7)
    x = (torch.randn(1, H, device = dev) * 0.5).half()
    e = 3

    # (a) wrapper reference
    u_a = mlp.ups[e].forward(x, {})
    g_a = mlp.gates[e].forward(x, {})
    a_a = ((g_a.float() / (1.0 + torch.exp(-g_a.float()))) * u_a.float()).half()
    d_a = mlp.downs[e].forward(a_a, {})
    print(f"(a) u[:4] {u_a.flatten()[:4].tolist()}", flush = True)
    print(f"(a) g[:4] {g_a.flatten()[:4].tolist()}", flush = True)
    print(f"(a) d absmax {d_a.abs().max().item():.4f} d[:4] {d_a.flatten()[:4].tolist()}", flush = True)

    # (b) manual reconstruct chain for the up projection only
    up = mlp.ups[e].inner
    w = torch.empty((H, I), dtype = torch.half, device = dev)
    ext.reconstruct(w, up.trellis, K, mcg, bool(up.mul1))
    h13 = torch.empty((1, H), dtype = torch.half, device = dev)
    ext.had_r_128(x, h13, up.suh, None, 1.0)
    gemm_out = torch.empty((1, I), dtype = torch.float, device = dev)
    ext.hgemm(h13, w, gemm_out)
    out_b = torch.empty((1, I), dtype = torch.half, device = dev)
    ext.had_r_128(gemm_out.half(), out_b, None, up.svh, 1.0)
    print(f"(b) u[:4] {out_b.flatten()[:4].tolist()}", flush = True)
    print(f"(a)==(b) up maxdiff {(u_a.half() - out_b).abs().max().item():.4e}", flush = True)

    # (v) host-side table + word expectations for e = 0..7 (match against kernel printf)
    for ep in range(8):
        gt = mlp.gates[ep].inner.trellis.view(torch.int32)
        print(f"v: e={ep} table_ptr {hex(int(mg.ptrs_trellis[ep].item()))} "
              f"direct_ptr {hex(int(gt.data_ptr()))} word0 {hex(int(gt[0,0,0].item()) & 0xffffffff)}", flush = True)

    # (w) direct production gemv vs (a), same expert, cfg as heuristic picks
    up_e = mlp.ups[e].inner
    A_had_w = torch.empty((1, H), dtype = torch.half, device = dev)
    C_w = torch.empty((1, I), dtype = torch.half, device = dev)
    ext.exl3_gemv(x, up_e.trellis, C_w, up_e.suh, A_had_w, up_e.svh, bool(up_e.mcg), bool(up_e.mul1))
    print(f"(w) exl3_gemv up maxdiff vs (a): {(C_w.flatten().float() - u_a.flatten().float()).abs().max().item():.4e}", flush = True)

    # (z) cross-match: does the kernel's expert e output match some OTHER expert's reference?
    ids32 = torch.tensor([0], dtype = torch.int32, device = dev)
    rows32 = torch.tensor([0], dtype = torch.int32, device = dev)
    rw = torch.ones(1, dtype = torch.half, device = dev)
    if True:
        n_try = 8
        refs = []
        for ep in range(n_try):
            refs.append(mlp.gates[ep].forward(x, {}).flatten().float())
        for e_z in range(n_try):
            ids_z = torch.tensor([e_z], dtype = torch.int32, device = dev)
            res = ext.p2b_stage_debug(x,
                mg.ptrs_trellis, mg.ptrs_suh, mg.ptrs_svh,
                mu.ptrs_trellis, mu.ptrs_suh, mu.ptrs_svh,
                md.ptrs_trellis, md.ptrs_suh, md.ptrs_svh,
                ids_z, rows32, rw, K, K, K, mcg, bool(mlp.multi_up.mul1), H, I, 2)
            gz = res[2][0].float()
            diffs = [(gz - refs[ep]).abs().max().item() for ep in range(n_try)]
            best = min(range(n_try), key = lambda i: diffs[i])
            print(f"z: kernel e={e_z} best-match ref e'={best} (maxdiff {diffs[best]:.4e}) "
                  f"own-diff {diffs[e_z]:.4e}", flush = True)

    # (c) p2b staged, weight 1.0, expert e
    for stage in (1, 2, 4):
        res = ext.p2b_stage_debug(x,
                      mg.ptrs_trellis, mg.ptrs_suh, mg.ptrs_svh,
                      mu.ptrs_trellis, mu.ptrs_suh, mu.ptrs_svh,
                      md.ptrs_trellis, md.ptrs_suh, md.ptrs_svh,
                      ids32, rows32, rw, K, K, K, mcg, bool(mlp.multi_up.mul1), H, I, stage)
        had_gate, had_up, gate, up, had_down, down, out_c = res
        if stage == 1:
            ref_had = torch.empty((1, H), dtype = torch.half, device = dev)
            ext.had_r_128(x, ref_had, mlp.gates[e].inner.suh, None, 1.0)
            print(f"S1 had_gate maxdiff vs python had_r_128: {(had_gate[0] - ref_had[0]).abs().max().item():.4e}", flush = True)
        if stage == 2:
            gf, kf = gate[0].float(), g_a.flatten().float()
            print(f"S2 gate maxdiff vs (a): {(gf - kf).abs().max().item():.4e}", flush = True)
            print(f"S2 up   maxdiff vs (a): {(up[0].float() - u_a.flatten().float()).abs().max().item():.4e}", flush = True)
            print(f"  gate k[:8]  {kf[:8].tolist()}", flush = True)
            print(f"  gate p2[:8] {gf[:8].tolist()}", flush = True)
            print(f"  gate p2[640-8:] {gf[-8:].tolist()}", flush = True)
            import torch as _t
            gs, ks = _t.sort(gf.abs()), _t.sort(kf.abs())
            print(f"  sorted-abs corr: {(gs.values - ks.values).abs().max().item():.4e} "
                  f"(0 = permutation/same set, large = different values)", flush = True)
            # per-128-chunk norms: is one chunk right and others wrong?
            for chunk in range(5):
                g_c = gf[chunk*128:(chunk+1)*128]; k_c = kf[chunk*128:(chunk+1)*128]
                print(f"  chunk {chunk}: k_absmax {k_c.abs().max().item():.4f} p2_absmax {g_c.abs().max().item():.4f}", flush = True)
        if stage == 4:
            print(f"S4 out  maxdiff vs (a): {(out_c[0].float() - d_a.flatten().float()).abs().max().item():.4e}", flush = True)
            print(f"(c) d[:4] {out_c.flatten()[:4].tolist()} absmax {out_c.abs().max().item():.4f}", flush = True)

    print("DISCRIM DONE", flush = True)

if __name__ == "__main__":
    main()
