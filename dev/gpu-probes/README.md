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
