"""Memory attribution: load the production-equivalent stack, dump per-device
allocated/reserved plus a gc-walk per-tensor breakdown grouped by size and source."""
import os, time, gc
os.environ["EXL3_NGRAM_STREAM"] = "1"
os.environ["EXL3_P2B_MOE"] = "1"

def main():
    import torch
    from exllamav3 import Config, Model, Cache, CacheLayer_fp16, Tokenizer
    from exllamav3.generator import Generator

    config = Config.from_directory("/models/exl3-4.05bpw")
    model = Model.from_config(config)
    draft = Model.from_config(config, component = "mtp")
    cache = Cache(model, max_num_tokens = 98304, max_batch_size = 2, max_history = 8)
    draft_cache = Cache(draft, max_num_tokens = 98304)
    for p in draft.load_gen(use_per_device = [22, 22, 22, 22], verbose = False):
        pass
    for p in model.load_gen(device = None, use_per_device = [22, 22, 22, 22],
                            tensor_p = True, tp_output_device = 0,
                            tp_backend = "native", verbose = False):
        pass
    print("loaded", flush = True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer,
                          draft_model = draft, draft_cache = draft_cache,
                          num_draft_tokens = 8, dynamic_draft_tokens = True,
                          draft_confidence = 0.6)
    print("generator built", flush = True)

    # per-device totals
    for d in range(4):
        alloc = torch.cuda.memory_allocated(d) / 1024**2
        resv = torch.cuda.memory_reserved(d) / 1024**2
        print(f"device {d}: allocated {alloc:.0f} MiB  reserved {resv:.0f} MiB", flush = True)

    # per-tensor gc walk on device 0 (the tight one), grouped by source
    groups = {}
    seen = set()
    for mod_src, mod in (("trunk", model), ("draft", draft)):
        for m in mod.modules:
            try:
                ts = m.get_tensors() or {}
            except Exception:
                continue
            for name, t in ts.items():
                try:
                    if torch.is_tensor(t) and t.is_cuda and t.device.index == 0:
                        key = f"{mod_src}:{type(m).__name__}"
                        groups[key] = groups.get(key, 0) + t.numel() * t.element_size()
                        seen.add(id(t.storage()))
                except Exception:
                    continue
    for name, t in vars(cache).items():
        try:
            if torch.is_tensor(t) and t.is_cuda and t.device.index == 0:
                groups[f"cache:{name}"] = groups.get(name, 0) + t.numel() * t.element_size()
        except Exception:
            continue
    # generic walk
    other = 0
    other_top = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.device.index == 0:
                n = obj.numel() * obj.element_size()
                if id(obj.storage()) not in seen:
                    other += n
                    other_top.append((n, tuple(obj.shape), str(obj.dtype)))
        except Exception:
            pass
    print("== device 0 named groups (MiB):", flush = True)
    for k, v in sorted(groups.items(), key = lambda x: -x[1])[:12]:
        print(f"  {v/1024**2:8.1f}  {k}", flush = True)
    print(f"  {other/1024**2:8.1f}  OTHER (unattributed tensors)", flush = True)
    other_top.sort(reverse = True)
    print("== top other tensors:", flush = True)
    for n, shape, dt in other_top[:10]:
        print(f"  {n/1024**2:8.1f} MiB  {shape} {dt}", flush = True)
    print("ATTR DONE", flush = True)

if __name__ == "__main__":
    main()
