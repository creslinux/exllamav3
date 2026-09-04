# GPU probe rigs (ice: 4x RTX 3090, TP4/layer-split qwen3.8-flash EXL3 4.05bpw)

Measurement harnesses used by the perf/tp4-qwen4exp work. Run inside the
side container against the tp4pkg overlay; every script loads the model
itself (~75 s) and prints its own result lines.

- tp4_bench.py         decode ms/step (warm+meas), BACKEND env
- tp4_ab_bench.py      matched-rules A/B (TP vs layer split), BENCH_TP env
- tp4_round_trace.py   per-round split: draft / verify(synced) / remainder
                       with rewind/realign/other sub-buckets
- tp4_probe_run.py     TRACE_STEP per-class verify decomposition and static
                       per-row scaling (MODE=decomp|scale, DRAFT_N env)
- tp4_submod_probe.py  attn/mlp/glue sub-module bracket per q_len
- tp4_mix_bracket.py   TP mixture internals: routing vs bc vs other
- tp4_sweep_clean.py   calibrator sweep, one confidence per process,
                       three measured runs with spread (CONF env)
- tp4_p2b_harness.py   p2b parity vs per-expert reference + timing vs
                       bc.run_bszN (bsz 1/2/4/8)
- tp4_discrim.py       p2b stage isolation (S1 hadamard / S2 gemv / S4
                       full), expert cross-match, production-gemv control

Standing measurement rules, each earned at least once:
warm job before every traced or measured run (first-seen shapes distort
the first window); sync every bracket before believing it (issuance is
not completion); three runs with spread when the effect is under ~10%;
one variable per process (the calibrator's learned state persists
across jobs and contaminates in-process sweeps).

Added later: tp4_round_trace_s (static-window pair -- the per-round-cost
instrument), tp4_conc (8-stream concurrency), tp4_longctx (deep-context
decode), tp4_prefill2 (cache-busted prefill gate), tp4_parity_mix
(mixture-level p2b vs bc), tp4_sweep_clean (one-conf-per-process),
tp4_os_parity (one-shot collective parity vs gloo -- harness v2; known to
hang at construction, see the one-shot commits), tp4_discrim (stage
isolation). Protocol rules added since the first commit: read every A/B
arm's completion before quoting its number; any new collective gets a
parity test before its first timing; per-round cost is measured at a
FIXED window; throughput batteries control the reasoning branch.
