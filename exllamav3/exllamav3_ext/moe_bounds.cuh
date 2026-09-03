#include <torch/extension.h>


#ifndef _moe_bounds_cuh_
#define _moe_bounds_cuh_

void moe_bounds
(
    at::Tensor counts,        // int64 [E], per-expert assignment counts (e.g. torch.bincount)
    at::Tensor out_meta,      // int32 [2]: [0] num_active (0 < c <= threshold), [1] num_overflow (c > threshold)
    at::Tensor out_triples,   // int32 [E * 3]: packed (expert_idx, row_start, row_count) for overflow
                              // experts in ascending expert order; only [0 .. num_overflow*3) valid
    int64_t threshold
);

#endif
