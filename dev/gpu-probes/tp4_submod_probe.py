"""Sub-module probe: which sub-module carries the verify per-position slope.
Wraps the inline output-device worker's TransformerBlocks: block total, attn sub, mlp sub,
each bracketed with CUDA events, bucketed by q_len. Static draft window via DRAFT_N.
Verdict rule: mlp (the mixture) carrying the q_len slope == mixture convicted;
attn (GDN recurrence / QSA) carrying it == recurrence.
"""
import os, time, json
os.environ["EXL3_NGRAM_STREAM"] = "1"

DRAFT_N = int(os.environ.get("DRAFT_N", "7"))
WARM = 40

def main():
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, ComboSampler, CacheLayer_fp16
    from exllamav3.generator import Generator, Job

    config = Config.from_directory("/models/exl3-4.05bpw")
    model = Model.from_config(config)
    draft = Model.from_config(config, component = "mtp")
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 4, max_history = 8)
    draft_cache = Cache(draft, max_num_tokens = 4096, layer_type = CacheLayer_fp16)
    for p in draft.load_gen(use_per_device = [22, 22, 22, 22], verbose = False):
        pass
    for p in model.load_gen(device = None, use_per_device = [22, 22, 22, 22],
                            tensor_p = True, tp_output_device = 0,
                            tp_backend = "native", verbose = False):
        pass
    print("loaded", flush = True)

    inline = model.mp_parent_conn[model.tp_output_device]
    mods = inline.local_context["modules"]
    events = []
    st = {"pass": 0, "rec": False}

    def ev():
        e = torch.cuda.Event(enable_timing = True)
        e.record()
        return e

    n_blocks = 0
    for m in mods:
        if type(m).__name__ != "TransformerBlock":
            continue
        n_blocks += 1
        block, attn, mlp = m, getattr(m, "attn", None), getattr(m, "mlp", None)
        orig_b = m.forward
        orig_a = attn.forward if attn is not None else None
        orig_m = mlp.forward if mlp is not None else None

        def make_b(orig_b):
            def f(x, params, *a, **kw):
                if params.get("prefill"):
                    return orig_b(x, params, *a, **kw)
                st["pass"] += 1
                st["rec"] = st["pass"] > WARM * n_blocks
                e0 = ev() if st["rec"] else None
                try:
                    return orig_b(x, params, *a, **kw)
                finally:
                    if st["rec"]:
                        events.append(("block", x.size(1), e0, ev()))
                    st["rec"] = False
            return f
        def make_sub(orig, cls):
            def f(x, params, *a, **kw):
                if not st["rec"] or orig is None:
                    return orig(x, params, *a, **kw)
                e0 = ev()
                r = orig(x, params, *a, **kw)
                events.append((cls, x.size(1), e0, ev()))
                return r
            return f

        m.forward = make_b(orig_b)
        if orig_a is not None:
            attn.forward = make_sub(orig_a, "attn")
        if orig_m is not None:
            mlp.forward = make_sub(orig_m, "mlp")
    print(f"wrapped {n_blocks} inline blocks", flush = True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer,
                          draft_model = draft, draft_cache = draft_cache,
                          num_draft_tokens = DRAFT_N, dynamic_draft_tokens = False,
                          record_draft_stats = True)
    sampler = ComboSampler(temperature = 0.6, top_k = 20, top_p = 0.95)
    req = json.load(open("/host/healthy_long.json"))
    ids = tokenizer.encode(tokenizer.hf_render_chat_template(req["messages"]))
    job = Job(input_ids = ids, max_new_tokens = 600, sampler = sampler, identifier = "probe")
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        generator.iterate()

    torch.cuda.synchronize()
    agg = {}
    counts = {}
    for cls, qlen, e0, e1 in events:
        k = (qlen, cls)
        agg[k] = agg.get(k, 0.0) + e0.elapsed_time(e1)
        counts[k] = counts.get(k, 0) + 1
    passes = counts.get((max(q for q, c in counts if c == "block"), "block"), 0) or st["pass"]
    print(f"DRAFT_N={DRAFT_N}  total block passes recorded basis: {st['pass']}", flush = True)
    for qlen in sorted({q for q, _ in agg}):
        np = counts.get((qlen, "block"), 1)
        row = []
        for cls in ("block", "attn", "mlp"):
            if (qlen, cls) in agg:
                row.append(f"{cls} {agg[(qlen, cls)]/np:7.3f}ms")
        if (qlen, "attn") in agg and (qlen, "mlp") in agg:
            glue = (agg[(qlen, "block")] - agg[(qlen, "attn")] - agg[(qlen, "mlp")]) / np
            row.append(f"glue {glue:6.3f}ms")
        print(f"  qlen={qlen} (x{np} passes): " + " | ".join(row), flush = True)
    stats = job.draft_stats
    tw = sum(s[1] for s in stats) or 1
    ta = sum(s[2] for s in stats)
    print(f"accept {100.0*ta/tw:.1f}% rounds {len(stats)}", flush = True)
    print("SUBMODULE PROBE DONE", flush = True)

if __name__ == "__main__":
    main()
