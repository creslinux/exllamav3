# Changelog

Work on this fork of [exllamav3](https://github.com/turboderp-org/exllamav3), serving
**Qwen3.8-Flash-Next EXL3 4.05bpw** on **4x RTX 3090** (PCIe Gen3, layer split) through
TabbyAPI, as a daily coding driver rather than a benchmark rig.

Forked from upstream `dev` at `843725c` (2026-09-02).

**Where things stand:** single-stream decode 108 tok/s, eight-stream aggregate 144.5 tok/s (up
from 98 before the batch limit was found),
prefill ~1,500 tok/s, and **three concurrent 260k-token sessions** in 96 GB of VRAM where the
stock configuration held one. Tool-calling quality is unchanged: 89/100 on tool-eval-bench
against an 88/100 baseline, both inside the benchmark's own +/- 2.5 noise.

Larger efforts are cited as a commit range rather than every hash; several commits appear more
than once because work was cherry-picked between the tensor-parallel and layer-split lines.

Every performance claim below was measured on this rig, before and after, on the same prompts.
Changes that did not survive their own measurement are recorded too, with the number that
killed them — several of them were confident theories.

---

## 2026-09-08 — Q8/Q6 quantised KV cache for the QSA attention layers

`e28549e` (merge), `dc4fe30` `2c35d4b` `3754f09` `ba324f8` `46f0dc6` `05737aa`
gates `2a30152` `5d6fac1` `8c63f94` `9ecb967` `a90db7e`

The KV cache for the 12 full-attention layers now stores keys and values quantised (Q4 through
Q8) while the QSA indexer's two side planes — the raw per-token keys and the pooled per-block
keys — stay fp16. That split is the whole design: block selection reads the same values it
always did, so the sparse attention picks the same blocks, and only the attended K/V carry
rounding error. A published fp8 KV patch for this model regressed a long-reasoning benchmark
from 6/6 to 2/6 precisely because it quantised the selector's keys as well.

The sparse attention kernel gained a packed-cache load path as a **separate** Triton kernel,
`_qsa_sparse_split_kernel_q`, leaving the fp16 kernel byte-identical so the graph-captured fp16
path and its C++ launch signature could not be disturbed. The first attempt inserted the new
arguments into the shared kernel and silently broke the captured fp16 signature; the separation
is the fix. The captured decode path was then armed for the quantised layer so decode keeps its
CUDA graphs instead of dropping to the eager kernel.

Six gates, all with an fp16 control:

| gate | result |
|---|---|
| block-selection divergence | layer 3, whose inputs cannot carry a quantised value, at parity (0.98x); overall 1.07x |
| end-of-prompt logits, 16k and 128k, prose and code | top-1 identical, Q8 inside the fp16 run-to-run spread |
| long-generation drift, 2k tokens | acceptance 62.5/83.5% fp16 vs 64.0/84.7% Q8, coherent tails |
| full-context 260k request | completes, zero errors |
| decode on the captured path | unchanged at 64k and above |
| storage, like for like | 720-960 MiB per card returned at a 262k pool |

Bit-identical block selection turned out to be unmeasurable: two identical fp16 runs already
disagree on 95.6% of query rows, because MoE atomic reductions make this stack
nondeterministic. The gate was re-scoped to the per-row symmetric difference of the selected
sets, where the quantised run sits 1.07x above the model's own noise, and the layer-3 parity
result carries the argument.

---

## 2026-09-08 — Deployment: the pool ceiling was a reservation, not the cache

Not a code change; a configuration finding that unlocked everything above.

Quantising the cache freed about 3 GB and bought **no extra context whatsoever**. Every attempt
to raise the pool past 524,288 tokens was refused with "Insufficient VRAM in split", at Q8 and
at Q6 alike, at every GPU split up to the physical limit of the cards.

The cause is `EXL3_QC_STAGING`, which defaults to 1. In that mode prefill dequantises into a
shared fp16 scratch **sized for the entire cache**, and the autosplit measuring pass triggers
its allocation, so the space is reserved at load time. That reservation is identical at fp16,
Q8 and Q6 — which is exactly why quantising the stored cache moved the ceiling by zero tokens.
Serving with `EXL3_QC_STAGING=0` moves the dequant into the kernels and the pool went straight
from 524,288 to 786,432 tokens. The same reservation is the likely cause of a short-context
decode penalty seen during gate 5, since sub-threshold prefill chunks paid a whole-pool dequant
that long contexts amortised away.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was also never set on the serving container,
though every test rig had it.

Live configuration: `cache_mode: Q6`, `cache_size: 786432`, `max_batch_size: 3`,
`draft_cache_mode: Q8`, `gpu_split: [17, 22.5, 22.5, 22.5]`, `sysmem_kv_cache: 10240`.
Verified with three concurrent 260,310-token requests completing together in 630 s, peak
21.2/22.5/23.3/21.4 GiB, no errors.

**One agent at full context became three.**

---

## 2026-09-07 — The concurrency limit was a configuration line

Not a code change, and it was worth more than several that were.

The serving configuration had `max_batch_size: 2`. Aggregate throughput therefore plateaued at
about 98 tok/s from two streams onward: at four and eight concurrent requests the wall simply
doubled and doubled again, because requests were being served two at a time and the rest
queued. Every concurrency measurement taken before this was measuring a queue.

It also meant the count-readback removal shipped the day before could not execute at all. That
path only runs when a forward carries more than eight rows, and two streams times four verify
rows is eight. The optimisation was correct, deployed, and dormant.

| streams | batch 2 | batch 4 | batch 8 |
|---|---:|---:|---:|
| 1 | 73.6 | 79.3 | 76.0 |
| 2 | 96.9 | 95.1 | 98.6 |
| 4 | 99.6 | 112.5 | 111.9 |
| 8 | 98.2 | 114.8 | **144.5** |

Each batch slot costs about 0.72 GB of GDN recurrent state across the four cards, allocated at
load whether or not the slot is used, so the ceiling is memory rather than preference. The
serve now runs three slots against a 786k pool, which is the shape that suits full-context
sessions; eight slots suits many small ones.

---

## 2026-09-06 — The fused-MoE count readback, and a prefill campaign that mostly said no

`6497d40` (the keeper), `51d9abe`, with the investigation in `876c22e` `64799dc` `7d5d110`
`5425df0` `b0b7181`

The MoE path read back per-expert token counts from the device on every layer to decide which
experts exceed the fused kernel's row cap. Each token contributes at most one row per expert,
so an expert can never hold more rows than the chunk has tokens: when the cap covers the chunk,
the readback, the Python overflow loop and the dequantise-then-GEMM fallback are all provably
unnecessary. One condition removes the lot. Measured **+9% aggregate throughput at eight
concurrent streams**, zero VRAM cost, no numerics change. The row cap became an environment
variable so it can be raised without a rebuild.

The rest of the campaign was elimination, and it is worth keeping because the same ideas will
come round again:

- **Raising the row cap** removed 97-99% of overflow events and moved the wall by nothing. The
  fallback's cost overlapped the fused kernel's GPU time.
- **Porting a purpose-built fat-expert kernel**, which is worth +20% on a bandwidth-poor
  machine, is not indicated here: the fused kernel handles very large experts at least as fast
  as the fallback does, so the load-imbalance theory it addresses is refuted on this hardware.
- **8192-token prefill chunks** run out of memory. 4096 is worth about +4% on realistic text
  and needs a bigger cap; parked.
- One measurement claimed a 30% prefill gain and was confounded: at that chunk size the very
  large experts had left the fused kernel entirely.

What the campaign did establish: the fused MoE region is 50-70% of prefill wall and runs at
about 22% of tensor-core peak on the small experts, against 60 TFLOPS for the fallback path on
the large ones in the same layer. A four-fold per-FLOP gap inside one layer is the standing
lead for anyone returning to prefill.

---

## 2026-09-06 — CPU KV second tier and the context ladder

`eeb0ed9` `2fb9bf9`

Cold KV pages spill to pinned host RAM and return on demand, proven byte-correct under layer
split: it pushes on eviction and restores on allocation. This is what makes a returning session
resume in about a second instead of re-reading its whole prompt at ~1,500 tok/s — for a 260k
context that is the difference between a second and three minutes.

Sized at 16 GB it took the host down: the kernel OOM killer stopped the server when two
concurrent 260k jobs grew the process past a 91 GB host that idles at 41 GB with a 39 GB n-gram
table resident. It runs at 10 GB, which parks roughly two and a half full contexts at Q6.

Also landed: prose and coding context ladders to 246k, which is where the flat ~1,500 tok/s
prefill and the 103-120 tok/s coding decode figures come from.

---

## 2026-09-04 — Draft-confidence calibration

`984824d`

The speculative decoder's calibrator targets a running product of acceptance probabilities and
cuts the draft window when it drops below a threshold. The default is 0.4. A sweep under matched
rules found the curve still climbing at 0.5 on both prose and code — 61.6/54.3 at 0.3, 61.8/61.7
at 0.4, 63.9/69.2 at 0.5 — and the serve now runs 0.6 through `EXL3_DRAFT_CONFIDENCE`.

Worth up to about +7 tok/s on code for a one-line configuration change, and it applies to both
backends. An earlier attempt to reason about the right value analytically was wrong, because the
threshold targets the running product rather than per-position acceptance; the sweep settled it.

---

## 2026-09-04 — Two layer-split placement bugs

`b49f8db`, and `479df7d` on `fix/ls-device-context`

Enabling the new MoE kernel under layer split crashed at load with an illegal memory access
during kernel autotune. The kernel was being launched while the current CUDA device was not the
device holding that layer's weights. Fixed by wrapping the launch in the layer's device context.

The same class of fault, found separately: the layer-split forward did not keep the current
device pinned to the module's device across the module loop, which is latent for anything that
allocates or launches without an explicit device guard.

---

## 2026-09-04 — Where the concurrency plateau actually comes from

`4de1277`, with `972d544` and `10f30a2` on the tensor-parallel branch

Aggregate throughput flattens as streams are added, and two plausible culprits were priced and
eliminated: the samplers, and the host synchronisation in the MoE routing count. Neither is the
cause. The plateau is expert-read scaling — each additional concurrent sequence activates more
unique experts per layer, so the weight traffic per step grows with the batch even though the
arithmetic per sequence does not.

That is a property of the architecture at this size, not an implementation defect, and it is why
the batch-size table above flattens between four and eight streams rather than continuing to
climb.

---

## 2026-09-03 — p2b cooperative fused-MoE kernel

`28361cb` `bad10c9` `5be809e` `1d17fda`
(originally `fb5cb0c` `fe09cfb` `74ee7a0` `a7a2068` `2be602e` on the tensor-parallel branch)

A slot-table cooperative mixture-of-experts kernel, ported from an MIT-licensed vLLM plugin
that had measured it against exllamav3's own path. It takes an explicit list of row-expert
slots rather than the dense form, which lets zero-weight slots be skipped, and it replaced the
stock MoE decode path behind `EXL3_P2B_MOE`.

**The decode round went from 50.3 ms to 34.6 ms, a 31% cut**, in two stages: the kernel with
zero-weight slot skipping took it to 45.1, and four follow-ons took it to 34.6 — staged scratch
buffers replacing seven allocations per layer per pass, the fp32 accumulator returned directly,
the slot map built in one kernel, and the shared expert through its own captured graph.

Parity failed for a week and the tile was read seven times. It was never wrong. The fault was
two lines of codebook dispatch: a truthiness test on a value that was never zero, so every
launch decoded with the wrong table. The tell had been in the log the whole time — two
supposedly different codebooks producing a bit-identical error to five digits. **Two identical
results from different inputs is a bug, not a confirmation.**

---

## 2026-09-03 — Diagnostic instrumentation, kept in-tree

`bdcb492` `2e193d4` `3c6bff0` `b2beac8` `0a8d3a2`

Most of the findings in this changelog came from purpose-built probes rather than a general
profiler, and they are committed so the next question does not start from nothing:

- `EXL3_TP_STUB_COLLECTIVES` replaces the collectives with stubs, giving an upper bound on
  compute by running the forward on garbage. It priced the collectives at 3.7 ms of a 21.4 ms
  step.
- `EXL3_TP_TRACE_STEP` brackets each module in the worker loop with both CUDA events and host
  timers, split by layer class. Comparing the two columns is what proved the decode step is
  GPU-paced rather than dispatch-bound, and retired a whole line of host-side work.
- The GPU probe rigs live under `dev/gpu-probes`: the batteries, the concurrency sweeps, the
  context ladders, and every gate used for the quantised cache.

The general profiler, by contrast, produced the reading that sent five changes into the bin. It
attributed time spent waiting on the GPU to host work, because the host dispatches during sync
waits.

---

## 2026-09-02 — Tensor-parallel port for qwen4_exp (archived, deployable by flag)

`perf/tp4-qwen4exp`, ~30 commits from `edbb063` to `800a35b`

A complete tensor-parallel implementation for this architecture: gated RMS norm carrying its
activation through the export round trip, the QSA indexer replicated per rank because a single
raw key head cannot be split, the PLE layer and n-gram table shared across ranks, multi-token
prediction drafting under tensor parallelism, and the architecture flag flipped.

It works, and it is faster: **decode 84.7 tok/s single-stream against 75 for layer split, and
157 aggregate at eight streams**, with the p2b compaction contributing most of that — under
tensor parallelism roughly three quarters of the expert slots on each rank are masked and were
still being traversed.

It is archived anyway. The per-card memory tax of the replicated embedding, indexer and head
broke the 256k KV pool, and context is worth more here than tokens per second.

Two upstream design choices were re-learned with numbers and then deleted again: re-enabling
NCCL broadcast and gather cost 7.7 ms per step against the shared-memory fallback, and routing
small all-reduces through the device-side ring cost 3.5 ms against the CPU-assisted path. One
kernel fix was kept — the accumulate stage added raw 16-byte chunks as `float4`, which corrupts
packed fp16 and bf16, so the device path had been silently wrong on exactly the payloads it
targets.

A one-shot all-reduce variant produced the fastest number the project ever printed, 114.7
tok/s. Its output was `Write a\nWrite a\n<|im_start|>`. **Read the completion before quoting the
number.**

---

## 2026-09-02 — Six host-side optimisations, one keeper

`af18dee` (kept) and, flag-off or reverted, `7dda39a` `3b8a859` `c534583` `66e780a` `20af46e`

A profiler said the host was starving the GPUs: 82% of generation-thread time in the verify
pass, GPU utilisation 8-11%. Six ideas were priced on that reading and five were built.

**Hoisting hot-path scratch allocations** in the hyper-connection mixer and the MoE router is
the one that survived: +3-6% median, kept, and later reintroduced 48 times over in a new kernel
before being hoisted again.

The others, with the numbers that killed them:

| change | result | lesson |
|---|---|---|
| draft-loop sync trimming | -13 to -17% | the sync rode mandatory GPU work; the break it removed was compute control |
| n-gram stage in front of MTP | -7% | the trained draft head beats a suffix automaton on exactly the repetitive text where the automaton fires |
| per-block CUDA graph capture | parity | removing ~500 eager dispatches moved nothing |
| lag-1 draft window | confounded | two variables moved at once; the table was uninterpretable |
| device-side expert bounds | parity to -10% | ~2 ms of Python against ~28 ms/layer of real GPU work |
| raising the multi-row MoE graph ceiling (`MAX_BSZN` 8 to 32) | identical, at a VRAM cost | the graph path was not the binding constraint at verify shapes |

Together they falsify the profile that motivated them. The host dispatches *during* sync waits,
so time spent waiting had been counted as time spent working. The constraint is GPU-side and
structural: roughly 500 tiny kernels per decode step, each on a ~25 microsecond floor. Only
fewer or fatter kernels help, which is what made the MoE kernel port the right next move.

---

## 2026-09-01 — Server that reported healthy and returned 503

`e5b4777` `1a29490` `a14ab22` — **merged upstream as #319 and #320**

One failed job latched the generator permanently: the health endpoint stayed green while every
subsequent request failed. Two separate faults. A recurrent-state slot leaked on the page
allocation path, so slots ran out; and a single job's failure propagated out of the iteration
loop, killing every other job in flight and leaving the generator in an unrecoverable state.

Failures are now contained per job and delivered to the owning request as an error result. The
fixes came with a fault-injection reproduction and a full test matrix, and went upstream.

They also left a trap worth naming, since it cost a full measurement window later: a reaped job
exits *cleanly*. A benchmark that does not read `iterate()`'s results sees a job "complete" in a
second and reports a record-breaking number. **Every rig reads the results.**

---

## Standing measurement rules

Each was earned once, expensively:

- Parity before timing. Warm a shape before measuring it. One configuration per process.
- Never repeat a prompt: the prefix cache turns a re-run into 47,000 tok/s of "prefill".
- Synchronise every CUDA bracket. Unsynchronised host wall time is not a cost.
- Print once on the first call of any new path. Four separate paths have been measured while
  silently never engaging.
- Read the completion before quoting the number.
- Control the reasoning branch: the same prompt swings 40% on whether the model thinks.
- Greedy is nondeterministic here, so there is no token-identity gate. Quality means acceptance
  ratios in band plus benchmark scores within noise.
- Two identical results from different inputs is a bug.
- Clean rebuild after any signature change: copied source timestamps are not honoured.
- Change one variable per measurement. Fallback paths must be tested, not assumed.

## Not yet done

- A 1,048,576-token pool is still refused with staging disabled, so a second reservation that
  scales with pool size remains unidentified. It is the gate to a fourth concurrent session.
- The GDN recurrent state is fp32 and costs about 0.7 GB per batch slot; in bf16 that halves.
- An NVMe third tier below the pinned-RAM one would park tens of contexts instead of two.
- Q6 shipped without its own numerics gates; the benchmark result is indirect evidence.
- One behavioural regression: on a missing required parameter the model now invents a query
  rather than asking, reproducibly in three of three trials.
