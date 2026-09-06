"""Coding-slanted ladder to 250k, protocol-correct: full warm pass (discarded), then measured.
Tail is an unterminated function so the model continues code, not answers."""
import json, urllib.request, random, sys

import os
KEY = os.environ["TABBY_API_KEY"]
BLOCKS = [
"""class Container{i}:
    \"\"\"Registry entry {i} for the shipping manifest service.\"\"\"
    def __init__(self, id: int, capacity: {c}):
        self.id = id
        self.capacity = capacity
        self.items: list[str] = []

    def add(self, sku: str, qty: int = 1) -> bool:
        if len(self.items) + qty > self.capacity:
            return False
        self.items.extend([sku] * qty)
        return True

    def manifest(self) -> dict:
        return {{"id": self.id, "count": len(self.items), "cap": self.capacity}}
""",
"""def resolve_dependency_{i}(graph: dict[str, list[str]], root: str, depth: int = {c}) -> list[str]:
    # Topological walk with cycle guard; depth-limited for large graphs.
    seen, order = set(), []
    def visit(node: str, d: int):
        if node in seen or d > depth:
            return
        seen.add(node)
        for dep in graph.get(node, []):
            visit(dep, d + 1)
        order.append(node)
    visit(root, 0)
    return order
""",
"""# Module: metrics_{i}.py -- collected {c} samples per window, exponential decay.
from dataclasses import dataclass

@dataclass
class Sample_{i}:
    ts: float
    value: float
    weight: float = 1.0

def decay(samples: list, half_life: float = 3600.0) -> float:
    if not samples:
        return 0.0
    total = sum(s.value * s.weight for s in samples)
    return total / max(1e-9, sum(s.weight for s in samples))
""",
]
TAIL = ("\n\n# Continue this codebase. Next up, the validation layer:\n\n"
        "def validate_manifest(manifest: dict, retries: int = 3) -> list[str]:\n"
        "    \"\"\"Validate a shipping manifest and return a list of violations.\n"
        "    Retries flaky lookups up to `retries` times with backoff.\n"
        "    \"\"\"\n")

def filler(blocks, seed):
    rng = random.Random(seed)
    out = []
    for i in range(blocks):
        out.append(BLOCKS[i % len(BLOCKS)].format(i=rng.randrange(10000), c=rng.randrange(8, 4096)))
    return "\n\n".join(out)

def bench(blocks, seed, max_tok):
    prompt = (f"// Repository slice {seed}: shipping-manifest service.\n\n" + filler(blocks, seed) + TAIL)
    payload = json.dumps({"model": "exl3-4.05bpw", "prompt": prompt, "max_tokens": max_tok,
                          "temperature": 0.6, "top_k": 20, "top_p": 0.95}).encode()
    req = urllib.request.Request("http://localhost:5000/v1/completions", data=payload,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())["usage"]

per_block = bench(40, 999, 16)["prompt_tokens"] / 40
print(f"calibration: {per_block:.0f} tok/block", flush=True)
TARGETS = [4000, 16000, 64000, 131000, 200000, 250000]
RUNGS = [(int(t / per_block), 128) for t in TARGETS]

phase = sys.argv[1]
if phase == "warm":
    for i, (blocks, mt) in enumerate(RUNGS):
        u = bench(blocks, 3000 + i, mt)
        print(f"warm {u['prompt_tokens']} tok, gen {u['completion_tokens']}", flush=True)
    print("WARM PASS DONE", flush=True)
else:
    print("ctx_tokens\tpptok\tpp_ts\ttgtok\ttg_ts", flush=True)
    results = []
    for i, (blocks, mt) in enumerate(RUNGS):
        u = bench(blocks, 2000 + i, mt)
        row = (u["prompt_tokens"], u["prompt_tokens_per_sec"], u["completion_tokens"], u["completion_tokens_per_sec"])
        results.append(row)
        print(f"{row[0]}\t{row[1]:.0f}\t{row[2]}\t{row[3]:.1f}", flush=True)
    json.dump(results, open("/tmp/opencode/ctx_bench_code_results.json", "w"))
    print("CODE BENCH DONE", flush=True)
