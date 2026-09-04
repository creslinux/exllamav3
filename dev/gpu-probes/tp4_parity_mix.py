"""Mixture-level parity: identical input through the bc path and the p2b path, one module,
flag flipped between calls. Covers the full mlp forward including the new shared-expert route."""
import os
os.environ["EXL3_NGRAM_STREAM"] = "1"
os.environ["EXL3_P2B_MOE"] = "0"   # module reads at import; flipped per-call below

def main():
    import torch
    from exllamav3 import Config, Model, Cache
    import exllamav3.modules.block_sparse_mlp as bsm

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
    assert mlp is not None
    dev, H, topk = mlp.device, mlp.hidden_size, mlp.num_experts_per_tok

    torch.manual_seed(99)
    for bsz in (1, 4, 8):
        x = (torch.randn(1, bsz, H, device = dev) * 0.5).half()
        params = {}
        # warm both paths (captures/configs settle)
        bsm._p2b_moe_env = False
        for _ in range(3): out_bc = mlp.forward(x, params)
        bsm._p2b_moe_env = True
        assert mlp._p2b_ok()
        for _ in range(3): out_p2b = mlp.forward(x, params)
        d = (out_p2b.float() - out_bc.float()).abs()
        print(f"bsz {bsz}: maxdiff {d.max().item():.4e}  absmax(bc) {out_bc.abs().max().item():.4f} "
              f"rel {(d.max() / out_bc.abs().max().clamp(min=1e-6)).item():.3e}", flush = True)
    bsm._p2b_moe_env = False
    print("PARITY MIXTURE DONE", flush = True)

if __name__ == "__main__":
    main()
