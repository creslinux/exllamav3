"""Phase 1 gate repro: serving-shaped LS load + MTP draft + generation with EXL3_P2B_MOE=1.

The standalone p2b harness (tp4_p2b_harness.py) never crashes because it calls
ext.p2b_fused_moe directly, bypassing the integrated dispatch in BlockSparseMLP.forward.
The serving path that crashed (battery arm ab3lp, twice: illegal memory access,
coop_autotune.cu:407) differs in: layer-split gpu_split load, MTP draft attach
(component="mtp"), draft autosplit, Generator with draft, first speculative generation.
This rig reproduces that shape minimally, with stage prints so a crash localizes to the
stage between the last two lines of output.

Arms:
  python ls_p2b_repro.py                 # suspect arm: EXL3_P2B_MOE=1, draft on, bsz 1
  EXL3_P2B_MOE=0 python ls_p2b_repro.py  # control: same load, integrated path off
  python ls_p2b_repro.py --bsz 8         # draft windows at higher batch
  python ls_p2b_repro.py --no-draft      # isolate the draft as the trigger
  python ls_p2b_repro.py --full-cache    # serving cache size (262144) vs battery 32768
"""
import argparse, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsz", type = int, default = 1)
    ap.add_argument("--no-draft", action = "store_true")
    ap.add_argument("--full-cache", action = "store_true")
    ap.add_argument("--new-tokens", type = int, default = 32)
    args = ap.parse_args()

    print(f"STAGE env: EXL3_P2B_MOE={os.environ.get('EXL3_P2B_MOE', '(unset)')} "
          f"args={vars(args)}", flush = True)

    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, ArgmaxSampler
    from exllamav3.generator import Generator, Job

    torch.cuda.set_device(0)

    model_dir = "/models/exl3-4.05bpw"
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)

    cache_size = 262144 if args.full_cache else 32768
    # max_history must cover the draft window for recurrent-state rewind (serving passes
    # draft_num_tokens; without it the first speculative verify faults on the state shape)
    max_history = 0 if args.no_draft else 6
    cache = Cache(model, max_num_tokens = cache_size, max_batch_size = 8,
                  max_history = max_history)

    print("STAGE loading main model (gpu_split [20,20,20,23])", flush = True)
    for p in model.load_gen(device = None, use_per_device = [20, 20, 20, 23],
                            max_batch_size = 8, verbose = False):
        pass
    print("STAGE main model loaded", flush = True)

    draft_model, draft_cache = None, None
    if not args.no_draft:
        print("STAGE loading MTP draft (autosplit, as serving does)", flush = True)
        draft_model = Model.from_config(config, component = "mtp")
        draft_cache = Cache(draft_model, max_num_tokens = cache_size, max_batch_size = 8,
                            max_history = max_history)
        for p in draft_model.load_gen(device = None, use_per_device = None,
                                      max_batch_size = 8, verbose = False):
            pass
        print("STAGE draft loaded", flush = True)

    tokenizer = Tokenizer(config)

    gen = Generator(
        model, cache,
        draft_model = draft_model,
        draft_cache = draft_cache,
        tokenizer = tokenizer,
        max_batch_size = 8,
        num_draft_tokens = 6 if draft_model is not None else None,
        dynamic_draft_tokens = True,
    )
    print("STAGE generator created", flush = True)

    prompt = ("You are a helpful assistant. Write a short paragraph about GPUs. "
              "Do not think; answer directly.") * args.bsz
    ids = tokenizer.encode(prompt)

    job = Job(
        ids,
        max_new_tokens = args.new_tokens,
        sampler = ArgmaxSampler(),
    )
    gen.enqueue(job)
    print("STAGE job enqueued; iterating", flush = True)
    while gen.num_remaining_jobs():
        gen.iterate()
    seq = job.sequences[0].sequence_ids
    all_t = seq.torch().view(-1)
    text = tokenizer.decode(all_t[-args.new_tokens:])
    print(f"STAGE generation done: {all_t.numel()} total tokens in sequence, "
          f"tail starts: {text[:80]!r}", flush = True)

    print("REPRO PASS", flush = True)

if __name__ == "__main__":
    main()
