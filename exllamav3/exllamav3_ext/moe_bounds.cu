#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "util.h"
#include "moe_bounds.cuh"

#define MOE_BOUNDS_MAX_E 1024
#define MOE_BOUNDS_THREADS 1024

/*
Device-side expert-bounds computation for the grouped MoE path.

Replaces the per-layer host readback of the full per-expert histogram
(513-int .tolist(), plus a 512-iteration Python loop that mostly skips)
with one single-block kernel that computes, entirely on device:

  - row_start[e]  (exclusive prefix sum of counts) -- packed into the
    overflow triples only, since the fused kernel reads counts directly
  - num_active:   experts with 0 < count <= threshold (the fused kernel's
    launch bound)
  - num_overflow: experts with count > threshold (the per-expert fallback
    loop's work list), packed as ascending (expert_idx, start, count)
    triples

The host then reads TWO ints (one small sync) and, in the common case of
zero overflow experts (assignment density << threshold), runs no further
readback and no Python expert loop at all.
*/
__global__ void moe_bounds_kernel
(
    const int64_t* __restrict__ counts,
    int E,
    int threshold,
    int* __restrict__ out_meta,        // [2]
    int* __restrict__ out_triples      // [E * 3]
)
{
    __shared__ int tmp[MOE_BOUNDS_THREADS];

    int tid = threadIdx.x;
    int c = (tid < E) ? (int)counts[tid] : 0;

    // Scan 1: exclusive prefix sum of counts -> row start offsets
    int inc_c = c;
    #pragma unroll
    for (int off = 1; off < MOE_BOUNDS_THREADS; off <<= 1)
    {
        __syncthreads();
        tmp[tid] = inc_c;
        __syncthreads();
        if (tid >= off) inc_c += tmp[tid - off];
    }
    int row_start = inc_c - c;

    // Scan 2: overflow flag (count > threshold) -> ordered compaction slots
    int is_over = (tid < E && c > threshold) ? 1 : 0;
    int inc_o = is_over;
    #pragma unroll
    for (int off = 1; off < MOE_BOUNDS_THREADS; off <<= 1)
    {
        __syncthreads();
        tmp[tid] = inc_o;
        __syncthreads();
        if (tid >= off) inc_o += tmp[tid - off];
    }
    if (is_over)
    {
        int slot = inc_o - 1;
        out_triples[slot * 3 + 0] = tid;
        out_triples[slot * 3 + 1] = row_start;
        out_triples[slot * 3 + 2] = c;
    }
    if (tid == MOE_BOUNDS_THREADS - 1)
        out_meta[1] = inc_o;

    // Scan 3: active flag (0 < count <= threshold) -> num_active
    int is_act = (tid < E && c > 0 && c <= threshold) ? 1 : 0;
    int inc_a = is_act;
    #pragma unroll
    for (int off = 1; off < MOE_BOUNDS_THREADS; off <<= 1)
    {
        __syncthreads();
        tmp[tid] = inc_a;
        __syncthreads();
        if (tid >= off) inc_a += tmp[tid - off];
    }
    if (tid == MOE_BOUNDS_THREADS - 1)
        out_meta[0] = inc_a;
}

void moe_bounds
(
    at::Tensor counts,
    at::Tensor out_meta,
    at::Tensor out_triples,
    int64_t threshold
)
{
    TORCH_CHECK(counts.is_cuda() && out_meta.is_cuda() && out_triples.is_cuda(),
                "moe_bounds: device tensors required");
    TORCH_CHECK_DTYPE(counts, kLong);
    TORCH_CHECK_DTYPE(out_meta, kInt);
    TORCH_CHECK_DTYPE(out_triples, kInt);
    int E = (int)counts.size(0);
    TORCH_CHECK(E <= MOE_BOUNDS_MAX_E, "moe_bounds: E > 1024 unsupported");
    TORCH_CHECK(out_meta.numel() == 2, "moe_bounds: out_meta must have 2 elements");
    TORCH_CHECK(out_triples.numel() >= (long)E * 3, "moe_bounds: out_triples too small");

    const at::cuda::OptionalCUDAGuard device_guard(counts.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    moe_bounds_kernel<<<1, MOE_BOUNDS_THREADS, 0, stream>>>(
        counts.data_ptr<int64_t>(),
        E,
        (int)threshold,
        out_meta.data_ptr<int>(),
        out_triples.data_ptr<int>()
    );
    C10_CUDA_CHECK(cudaPeekAtLastError());
}
