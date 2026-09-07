import torch
fp1 = torch.load('/tmp/opencode/gate1_idx_fp16_r1_list.pt')
fp2 = torch.load('/tmp/opencode/gate1_idx_fp16_r2_list.pt')
q8  = torch.load('/tmp/opencode/gate1_idx_q8_list.pt')
N = min(len(fp1), len(fp2), len(q8))   # shared prefill prefix (fp16 decode went via BC)
print(f"tensors: fp16_r1 {len(fp1)} fp16_r2 {len(fp2)} q8 {len(q8)} -> shared {N}")

def symdiff_mean(A, B):
    A = torch.sort(A, dim=1).values
    B = torch.sort(B, dim=1).values
    W = A.shape[1]
    iAB = torch.searchsorted(B, A).clamp(max=W-1)
    inB = (torch.searchsorted(B, A) < W) & (B.gather(1, iAB) == A)
    iBA = torch.searchsorted(A, B).clamp(max=W-1)
    inA = (torch.searchsorted(A, B) < W) & (A.gather(1, iBA) == B)
    return ((~inB).sum(dim=1) + (~inA).sum(dim=1)).float().mean().item()

LAYERS = ["L3","L7","L11","L15","L19","L23","L27","L31","L35","L39","L43","L47"]
self_by_layer = [0.0]*12
q8_by_layer   = [0.0]*12
cnt = [0]*12
for t in range(N):
    ly = t % 12
    a, b, c = fp1[t], fp2[t], q8[t]
    if a.shape != b.shape or a.shape != c.shape:
        print(f"tensor {t} shape mismatch: {a.shape} vs {b.shape} vs {c.shape}")
        continue
    self_by_layer[ly] += symdiff_mean(a, b)
    q8_by_layer[ly]   += symdiff_mean(a, c)
    cnt[ly] += 1

print(f"\n{'layer':6} {'fp16 self':>10} {'q8-vs':>10} {'ratio':>6}")
for i in range(12):
    s = self_by_layer[i]/max(1,cnt[i]); q = q8_by_layer[i]/max(1,cnt[i])
    flag = "  <- first attn layer" if i == 0 else ""
    print(f"{LAYERS[i]:6} {s:10.1f} {q:10.1f} {q/max(s,1e-9):6.2f}x{flag}")

# overall
S = sum(self_by_layer)/sum(cnt); Q = sum(q8_by_layer)/sum(cnt)
print(f"\noverall: fp16 self {S:.1f}, q8-vs {Q:.1f}, ratio {Q/S:.2f}x")
