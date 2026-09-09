# Qwen3.8-Flash-Next on 4x RTX 3090 — complete recipe

What this builds: a TabbyAPI server on four consumer 3090s running
Qwen3.8-Flash-Next at EXL3 4.05bpw, layer split, with multi-token-prediction
drafting and a quantised KV cache.

Measured on the reference machine, not projected:

| | |
|---|---:|
| decode, single stream | 108 tok/s |
| decode, 8 concurrent | 144.5 tok/s aggregate |
| prefill | ~1,500 tok/s, flat to 246k |
| concurrent 260k-token sessions | 3 |
| tool-calling quality (tool-eval-bench, 84 scenarios) | 89/100 |

The last row matters more than the first: quality is unchanged from an fp16
cache, inside the benchmark's own noise.

---

## 1. Hardware and host

| | reference machine | minimum |
|---|---|---|
| GPUs | 4x RTX 3090, 24 GB each | same; 96 GB total is the design point |
| slots | Gen3, mixed x8/x16 | anything; see the note below |
| host RAM | 91 GB | 64 GB, and it will be tight |
| disk | NVMe | ~110 GB free for the model |
| driver | 610.57.04 | any driver supporting CUDA 13 |

**You do not need NVLink, and you do not need a P2P-patched driver.** Layer
split runs one device at a time and moves a hidden state across the boundary,
which is kilobytes per decode step. We measured peer-to-peer at exactly zero
benefit at decode payloads. Slot width does not matter much either, for the
same reason. Skip that whole rabbit hole.

**Host RAM is the constraint people underestimate.** The model's n-gram table
is 36 GB and is held resident (`ngram_ram: true`), because that is worth a
large prefill margin over streaming it from disk. With the pinned KV tier
enabled the process idles around 51 GB. On a 64 GB host, disable the tier.

---

## 2. The model

From `turboderp/Qwen3.8-Flash-Next-exl3`, branch `4.05bpw_h6_ng6`:

```bash
huggingface-cli download turboderp/Qwen3.8-Flash-Next-exl3 \
  --revision 4.05bpw_h6_ng6 --local-dir /models/exl3-4.05bpw
```

About 104 GB: 63 GB of trunk shards, 36 GB n-gram table, 0.5 GB vision tower.

Other branches exist at 2.05, 3.05, 5.05 and 6.05 bpw. 4.05 is the sweet spot
on 96 GB of VRAM; 5.05 leaves too little for a useful KV pool.

### Optional: a 6-bit draft head

The 4.05bpw branch ships a 4-bit multi-token-prediction draft. Splicing the
6-bit draft from the `6.05bpw_h6_ng6` branch raises acceptance for about
600 MB on one card. The MTP tensors live in the last shard of each branch, and
their byte offsets are in the safetensors header, so only those ranges need
fetching.

**A trap that has caught us twice:** after splicing, `config.json` still says
`mtp_bits: 4`. It is stale metadata, not the truth. Confirm what you are
actually running by reading the trellis width from the shard header rather than
trusting the config. A 6-bit draft shows K=6 on the MTP expert tensors; 4-bit
shows K=4.

---

## 3. The engine

Three layers, and only the third is unusual.

**Base image.** CUDA 13 runtime, Python 3.12, a venv, TabbyAPI. The reference
machine builds from `nvidia/cuda:13.2.1-runtime-ubuntu24.04`. Any base works
provided the ABI matches the exllamav3 wheel: **torch 2.11.0+cu130 and
Python 3.12**.

**exllamav3 1.4.6**, the upstream release wheel:

```
pip install --no-deps \
  https://github.com/turboderp-org/exllamav3/releases/download/v1.4.6/exllamav3-1.4.6+cu132.torch2.11.0-cp312-cp312-linux_x86_64.whl
```

`--no-deps` matters. Let it resolve dependencies and it will replace your torch.

**This fork's package**, branch `perf/ls-opt`, bind-mounted over the installed
one. That branch carries the mixture-of-experts kernel, the allocator hoists,
the fused-count guard and the quantised QSA cache. See `CHANGELOG.md` for what
each of those is and what it measured.

### Building the extension

**This is the step that delivers the kernel.** The Python files are a bind
mount, but the mixture-of-experts kernel, the quantised-cache attention path
and the graph-capture bindings are CUDA and C++ that must be compiled from this
branch. Without this step you get the stock engine with none of it.

```bash
git clone -b perf/ls-opt https://github.com/creslinux/exllamav3.git
cd exllamav3
export TORCH_CUDA_ARCH_LIST="8.6"        # 3090s
export MAX_JOBS=8                        # nvcc is memory-hungry; tune to your RAM
python setup.py build_ext --inplace
```

`setup.py` walks the whole extension tree and compiles every `.c`, `.cpp` and
`.cu` it finds, so the new kernels are picked up with no build-file edits. It
produces `exllamav3_ext.cpython-312-x86_64-linux-gnu.so`, which is what the
compose file bind-mounts over the wheel's own extension.

Confirm the kernel actually made it in before deploying:

```bash
strings exllamav3_ext.cpython-312-*.so | grep -c p2b_fused_moe   # must be > 0
```

Two things will bite you:

1. **The serving image has no `nvcc`.** It runs prebuilt wheels. A working
   toolchain can be assembled from pip, and every piece must match torch's CUDA
   build:

   ```
   pip install nvidia-cuda-nvcc==13.2.86 nvidia-nvvm==13.2.86 \
               nvidia-cuda-cccl==13.2.86 nvidia-cuda-runtime==13.2.86 \
               nvidia-cuda-crt==13.2.86
   ln -sf .../nvidia/cu13/lib/libcudart.so.13 .../nvidia/cu13/lib/libcudart.so
   export CUDA_HOME=.../nvidia/cu13 && export PATH=$CUDA_HOME/bin:$PATH
   ```

   Verify the binaries rather than the pip metadata: `nvcc --version`,
   `ptxas --version` and the `CUDART_VERSION` in `cuda_runtime_api.h` must all
   agree. Pip metadata has lied about this.

2. **Build in a throwaway container, not the serving one.** With the model
   resident the serving container's cgroup has no headroom and the compile is
   OOM-killed. Mount the checkout into a fresh container off the same image.

If you copy sources into a container rather than mounting them, force a clean
rebuild: `docker cp` does not preserve timestamps, so incremental builds
silently skip changed translation units. That has produced two "the change did
nothing" results here that were really "the change was never compiled".

### What you get, and how to prove it is live

| feature | where it lives | proof it is active |
|---|---|---|
| p2b mixture-of-experts kernel | compiled `.so` | `EXL3_P2B_MOE=1` set, and `p2b_fused_moe` in the `.so` |
| quantised QSA KV cache | Python + `.so` | `cache_mode: Q6` loads without the fp16-only assertion |
| captured decode for the quant cache | `.so` | no eager fallback warning at load |
| fused-count guard, allocator hoists | Python | in the bind-mounted package |

The engine prints once on the first call of the quantised sparse path. If you
never see that line at long context, the path is not engaging.

---

## 4. Configuration

`deploy/config.example.yml` and `deploy/docker-compose.example.yml` in this
repository are the live files, annotated. Two environment flags in the compose
file are doing the heavy lifting and neither is on by default:

```
EXL3_QC_STAGING=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The first is the important one. Its default reserves an fp16 scratch buffer
**sized for the entire KV cache**, allocated during the load-time measuring
pass. That reservation is identical whether the cache is fp16, Q8 or Q6 — so
quantising the cache without this flag buys exactly zero extra context. It cost
us a night to find. Setting it to 0 moves the dequantisation into the kernels;
we measured no cost at long context.

The second stops the allocator fragmenting during load and reporting out of
memory well before the cards are full.

---

## 5. Two profiles

The pool and the slot count trade against each other. Pick by workload.

**Long sessions** — three agents at full context:

```yaml
cache_mode: Q6
cache_size: 786432      # 3 x 262144
max_batch_size: 3
draft_cache_mode: Q8
gpu_split: [17, 22.5, 22.5, 22.5]
```

**Many short sessions** — eight streams, best aggregate throughput:

```yaml
cache_mode: Q8
cache_size: 524288
max_batch_size: 8
```

The pool is shared across running requests, not per request. A request reserves
its prompt plus `max_tokens` up front; if the pool cannot hold it, it queues
rather than failing. Each batch slot costs about 0.72 GB of recurrent state
across the cards, allocated at load whether used or not.

---

## 6. Verify

In order, because each catches a different failure:

1. Model loads without "Insufficient VRAM in split". That message is a real
   allocation failure, not an estimate — the loader allocates and advances
   devices on out-of-memory. Raise `gpu_split` in ~1 GB steps.
2. `curl /v1/completions` with "The capital of France is" returns " Paris".
3. One request at your full context completes with no errors, and
   `nvidia-smi` still shows headroom on every card afterwards.
4. Concurrency: send N full-context requests simultaneously and confirm they
   complete together rather than serially.

Expected on a working build: single-stream decode 100-110 tok/s on code,
prefill 1,400-1,600 tok/s, draft acceptance 60-85% depending on content.

---

## 7. Three ways it has actually broken

**"Insufficient VRAM in split" at a larger pool.** Almost certainly the staging
reservation above. Set `EXL3_QC_STAGING=0` before assuming the cards are full.

**The kernel OOM-killer stopping the server.** The pinned host KV tier
(`sysmem_kv_cache`) is sized in MiB of **host** RAM, and it grows under load. At
16384 on a 91 GB host, two concurrent full-context jobs took the whole process
down. 10240 is the tested value here. On a 64 GB host, set it to 0.

**A benchmark reporting an impossible number.** Failed jobs are reaped and
delivered to the owning request as an error, so a job that dies exits *cleanly*.
Any harness that does not read the results will report a record-breaking run.
Read `iterate()`'s results, and read the completion before quoting a number.

---

## 8. What we tried that you can skip

- Tensor parallelism. It works and is faster single-stream, but the per-card
  memory cost of the replicated embedding, indexer and head breaks the KV pool.
  Context is worth more than tokens per second on 24 GB cards.
- P2P driver patching, and worrying about slot width. Zero benefit at decode.
- Prefill chunk sizes above 4096. Runs out of memory; 8192 is not viable here.
- CUDA graph capture at block scope, host-side synchronisation trimming, and
  n-gram drafting in front of the trained draft head. All measured parity or
  worse. `CHANGELOG.md` has the numbers that killed each one.
