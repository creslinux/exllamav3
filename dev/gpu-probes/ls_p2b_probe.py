"""Phase 1 discriminator: per-layer direct p2b call, on the layer's own device.

If every layer passes a direct ext.p2b_fused_moe call under an explicit device context,
the kernel and per-layer tables are sound and the serving-path fault is a launch-context
bug (kernel launched from the wrong current device). If a specific layer faults here too,
it is layer-local (tables/statics on that layer).

Run after a warm load; prints one PASS/FAIL line per MoE layer with its device.
"""
import torch
import exllamav3_ext as ext
from exllamav3 import Config, Model

def main():
    config = Config.from_directory("/models/exl3-4.05bpw")
    model = Model.from_config(config)
    for p in model.load_gen(device = None, use_per_device = [20, 20, 20, 23], verbose = False):
        pass
    print("loaded", flush = True)

    mlps = []
    for m in model.modules:
        if type(m).__name__ == "TransformerBlock" and getattr(m, "mlp", None) is not None \
                and type(m.mlp).__name__ == "BlockSparseMLP" and m.mlp.bc is not None:
            mlps.append(m.mlp)
    print(f"{len(mlps)} MoE layers", flush = True)

    bsz = 7   # verify-shaped row count
    fails = 0
    for mlp in mlps:
        mg, mu, md = mlp.multi_gate, mlp.multi_up, mlp.multi_down
        H, E, topk = mlp.hidden_size, mlp.num_local_experts, mlp.num_experts_per_tok
        I = mlp.intermediate_size_padded
        K = mg.K
        dev = mlp.device

        torch.manual_seed(42)
        with torch.cuda.device(dev):
            y = (torch.randn(bsz, H, device = dev) * 0.5).half()
            logits = torch.randn(bsz, E, device = dev)
            w, ids_g = torch.topk(logits, topk, dim = -1)
            w = torch.softmax(w.float(), dim = -1).half()
            rows = torch.arange(bsz, device = dev).repeat_interleave(topk).to(torch.int32)
            ids32 = ids_g.reshape(-1).to(torch.int32).contiguous()
            rw = w.reshape(-1).half().contiguous()
            out = torch.empty((bsz, H), dtype = torch.float, device = dev)
            n = bsz * topk
            gate = torch.empty((n, I), dtype = torch.half, device = dev)
            up = torch.empty((n, I), dtype = torch.half, device = dev)
            down = torch.empty((n, H), dtype = torch.half, device = dev)
            hg = torch.empty((n, H), dtype = torch.half, device = dev)
            hu = torch.empty((n, H), dtype = torch.half, device = dev)
            hd = torch.empty((n, I), dtype = torch.half, device = dev)
            acc = torch.empty((bsz, H), dtype = torch.float, device = dev)
            try:
                ext.p2b_fused_moe(
                    y, out,
                    mg.ptrs_trellis, mg.ptrs_suh, mg.ptrs_svh,
                    mu.ptrs_trellis, mu.ptrs_suh, mu.ptrs_svh,
                    md.ptrs_trellis, md.ptrs_suh, md.ptrs_svh,
                    ids32, rows, rw, K, K, K,
                    bool(mg.mcg), bool(mu.mul1), H, I,
                    gate, up, down, hg, hu, hd, acc,
                )
                torch.cuda.synchronize(dev)
                print(f"PASS {mlp.key} dev {dev}", flush = True)
            except Exception as e:
                fails += 1
                print(f"FAIL {mlp.key} dev {dev}: {e}", flush = True)
                break

    print(f"DONE fails={fails}", flush = True)

if __name__ == "__main__":
    main()
