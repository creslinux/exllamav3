import torch
K = 2080
fp1 = torch.load('/tmp/opencode/gate1_idx_fp16_run1.pt').reshape(-1, K)
fp2 = torch.load('/tmp/opencode/gate1_idx_fp16.pt').reshape(-1, K)
q8  = torch.load('/tmp/opencode/gate1_idx_q8.pt').reshape(-1, K)
n = fp1.shape[0]

def symdiff(A, B):
    # A, B: (rows, W) int32 -> symmetric difference cardinality per row (set semantics, -1s cancel)
    W = A.shape[1]
    iAB = torch.searchsorted(B, A)
    iABc = iAB.clamp(max=W-1)
    inB = (iAB < W) & (B.gather(1, iABc) == A)
    A_notB = (~inB).sum(dim=1)
    iBA = torch.searchsorted(A, B)
    iBAc = iBA.clamp(max=W-1)
    inA = (iBA < W) & (A.gather(1, iBAc) == B)
    B_notA = (~inA).sum(dim=1)
    return A_notB + B_notA

def report(name, A, B):
    A = torch.sort(A, dim=1).values
    B = torch.sort(B, dim=1).values
    sd = symdiff(A[:n], B[:n]).float()
    mean = sd.mean().item()
    p95 = sd.quantile(0.95).item()
    p50 = sd.median().item()
    p99 = sd.quantile(0.99).item()
    print(f"{name}: mean {mean:.2f}  median {p50:.2f}  p95 {p95:.2f}  p99 {p99:.2f}")
    return sd

fp16_self = report("fp16 self (r1 vs r2)", fp1, fp2)
q8_vs     = report("q8 vs fp16 (r1)", fp1, q8)
print(f"mean ratio q8/fp16: {q8_vs.mean().item()/fp16_self.mean().item():.2f}x")
print(f"p95 ratio  q8/fp16: {q8_vs.quantile(0.95).item()/fp16_self.quantile(0.95).item():.2f}x")

# bimodality / garbage check: histogram of the q8 symdiff counts
h = torch.bincount(q8_vs.long(), minlength=64)[:64]
big = (q8_vs > 32).sum().item()
print(f"q8 rows with symdiff > 32 (possible stale garbage): {big} / {n} = {100*big/n:.3f}%")
print("top histogram buckets (count, value):", [(int(c), i) for i, c in enumerate(h) if c > 0][:12])
