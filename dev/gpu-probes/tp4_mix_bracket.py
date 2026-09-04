"""TP mixture-internals bracket: routing chain vs bc graph (routed+shared) vs other,
inside the inline worker's BlockSparseMLP, bucketed by q_len. Static draft (DRAFT_N env).
Also prices the shared expert standalone on one module after generation."""
import os, time, json
os.environ["EXL3_NGRAM_STREAM"] = "1"

DRAFT_N = int(os.environ.get("DRAFT_N", "1"))
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
    st = {"pass": 0, "rec": False, "qlen": 0}

    def ev():
        e = torch.cuda.Event(enable_timing = True)
        e.record()
        return e

    class _BCProxy:
        __slots__ = ("_bc",)
        def __init__(self, bc):
            self._bc = bc
        def __getattr__(self, name):
            return getattr(self._bc, name)
        def run_bszN(self, y, ids, w):
            if not st["rec"]:
                return self._bc.run_bszN(y, ids, w)
            e0 = ev()
            r = self._bc.run_bszN(y, ids, w)
            events.append(("bc", st["qlen"], e0, ev()))
            return r

    n_mlp = 0
    sample_mlp = None
    for m in mods:
        if type(m).__name__ != "TransformerBlock":
            continue
        mlp = getattr(m, "mlp", None)
        if mlp is None or type(mlp).__name__ != "BlockSparseMLP":
            continue
        n_mlp += 1
        if sample_mlp is None and mlp.bc is not None:
            sample_mlp = mlp
        orig_f, orig_r, orig_b = mlp.forward, mlp.routing_fn, mlp.bc.run_bszN

        def wrap_total(orig_f):
            def f(x, params, *a, **kw):
                if params.get("prefill"):
                    return orig_f(x, params, *a, **kw)
                st["pass"] += 1
                st["rec"] = st["pass"] > WARM * n_mlp
                e0 = ev() if st["rec"] else None
                try:
                    return orig_f(x, params, *a, **kw)
                finally:
                    if st["rec"]:
                        events.append(("total", st["qlen"], e0, ev()))
                    st["rec"] = False
            return f
        # qlen from the routing input is unreliable here; capture from routing wrapper instead
        def wrap_routing(orig_r, mlp=mlp):
            def f(bsz, cfg, y, params):
                if not st["rec"]:
                    return orig_r(bsz, cfg, y, params)
                st["qlen"] = y.size(0)
                e0 = ev()
                r = orig_r(bsz, cfg, y, params)
                events.append(("routing", y.size(0), e0, ev()))
                return r
            return f

        mlp.forward = wrap_total(orig_f)
        mlp.routing_fn = wrap_routing(orig_r)
        mlp.bc = _BCProxy(mlp.bc)
    print(f"wrapped {n_mlp} TP mlps", flush = True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer,
                          draft_model = draft, draft_cache = draft_cache,
                          num_draft_tokens = DRAFT_N, dynamic_draft_tokens = False,
                          record_draft_stats = True)
    sampler = ComboSampler(temperature = 0.6, top_k = 20, top_p = 0.95)
    req = json.load(open("/host/healthy_long.json"))
    ids = tokenizer.encode(tokenizer.hf_render_chat_template(req["messages"]))
    job = Job(input_ids = ids, max_new_tokens = 600, sampler = sampler, identifier = "br")
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        generator.iterate()

    torch.cuda.synchronize()
    agg, counts = {}, {}
    for cls, qlen, e0, e1 in events:
        k = (qlen, cls)
        agg[k] = agg.get(k, 0.0) + e0.elapsed_time(e1)
        counts[k] = counts.get(k, 0) + 1
    for qlen in sorted({q for q, _ in agg}):
        np = counts.get((qlen, "total"), 1)
        row = [f"{cls} {agg[(qlen, cls)]/np:7.3f}ms" for cls in ("total", "routing", "bc") if (qlen, cls) in agg]
        other = None
        if all((qlen, c) in agg for c in ("total", "routing", "bc")):
            other = (agg[(qlen, "total")] - agg[(qlen, "routing")] - agg[(qlen, "bc")]) / np
            row.append(f"other {other:6.3f}ms")
        print(f"  qlen={qlen} (x{np}): " + " | ".join(row), flush = True)

    # Standalone pricing: shared expert + bc on the sample module
    if sample_mlp is not None:
        mlp = sample_mlp
        dev = mlp.device
        H, E, topk = mlp.hidden_size, mlp.num_experts, mlp.num_experts_per_tok
        for bsz in (2, 8):
            y = (torch.randn(bsz, H, device = dev) * 0.5).half()
            logits = torch.randn(bsz, E, device = dev)
            w, ids_g = torch.topk(logits, topk, dim = -1)
            w = torch.softmax(w.float(), dim = -1).half()
            sh = mlp.shared_experts
            def bench(fn, iters = 200, warm = 30):
                for _ in range(warm): fn()
                s = torch.cuda.Event(enable_timing = True); e = torch.cuda.Event(enable_timing = True)
                torch.cuda.synchronize(); s.record()
                for _ in range(iters): fn()
                e.record(); torch.cuda.synchronize()
                return s.elapsed_time(e) / iters * 1000
            real_bc = mlp.bc._bc if isinstance(mlp.bc, _BCProxy) else mlp.bc
            t_bc = bench(lambda: real_bc.run_bszN(y, ids_g, w))
            t_sh = float("nan")
            if sh is not None and sh.bc is not None:
                y3 = y.view(1, bsz, H)
                d3 = torch.empty_like(y3, dtype = torch.float)
                t_sh = bench(lambda: sh.bc.run_bszN(y3, d3))
            print(f"  standalone bsz {bsz}: bc(routed+shared) {t_bc:7.1f} us | shared-expert bc {t_sh:7.1f} us", flush = True)

    print("TP MIXTURE BRACKET DONE", flush = True)

if __name__ == "__main__":
    main()
