import json, urllib.request, random

KEY = "1bc4e26f7ef7dab3390e5a2261bd76da"
SENT = "The quarterly report covers depot logistics, regional staffing rotations, and deferred maintenance schedules for the period. "

def bench(words, seed, max_tok):
    body = SENT * (words // 18 + 2)
    prompt = (f"Document {seed} follows.\n\n" + body[:words * 7] +
              "\n\nContinue this document with a detailed elaboration of the findings:")
    payload = json.dumps({"model": "exl3-4.05bpw", "prompt": prompt, "max_tokens": max_tok,
                          "temperature": 0.6, "top_k": 20, "top_p": 0.95}).encode()
    req = urllib.request.Request("http://localhost:5000/v1/completions", data=payload,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["usage"]

bench(200, 999, 16)
print("ctx_tokens\tpptok\tpp_ts\ttgtok\ttg_ts", flush=True)
LADDER = [(100, 96), (500, 96), (1000, 96), (2000, 96), (4000, 96), (8000, 96),
          (16000, 128), (32000, 128), (65000, 128), (98000, 128), (104000, 128)]
results = []
for i, (words, mt) in enumerate(LADDER):
    u = bench(words, 1000 + i, mt)
    row = (u["prompt_tokens"], u["prompt_tokens_per_sec"], u["completion_tokens"], u["completion_tokens_per_sec"])
    results.append(row)
    print(f"{row[0]}\t{row[1]:.0f}\t{row[2]}\t{row[3]:.1f}", flush=True)
json.dump(results, open("/tmp/opencode/ctx_bench_results.json", "w"))
print("BENCH DONE", flush=True)
