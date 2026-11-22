"""RULER-4k evaluation of query-aware KV eviction at a fixed keep budget.
Baselines use kvpress presses exactly as kvpress/evaluation (query_aware=True). 'relhead' = proposed
method: relevance heads score text chunks; the same chunk set is kept in every layer/head."""
import argparse, json, os, random, time, ast
import numpy as np, torch, pandas as pd
from datasets import load_dataset
from transformers import DynamicCache, pipeline
from kvpress import SnapKVPress, AdaKVPress, PyramidKVPress, ChunkKVPress, CriticalAdaKVPress
from kvpress.utils import compute_n_kept
from common import (load_model, build_prompt, tokenize_with_offsets, QAttnCapture, heads_to_dict,
                    token_chunks, prune_cache)
import sys; sys.path.insert(0, "/mnt/beegfs/mkim54/relevance-kv/ext/kvpress/evaluation")
from benchmarks.ruler.calculate_metrics import calculate_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama3.1-8b")
ap.add_argument("--method", required=True)  # full|snapkv|adakv|pyramidkv|chunkkv|criticaladakv|relhead
ap.add_argument("--ratio", type=float, default=0.9)  # fraction evicted (0.9 = keep 10%)
ap.add_argument("--split", default="test")  # dev (20/task) | test (100/task)
ap.add_argument("--n_dev", type=int, default=20)
ap.add_argument("--n_test", type=int, default=100)
ap.add_argument("--tasks", default="all")
# relhead options
ap.add_argument("--heads_file")
ap.add_argument("--heads_mode", default="top")  # top | random | all | bottom
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--agg", default="density")  # head ranking criterion from heads_file + chunk score: density|mass
ap.add_argument("--chunk", default="sentence")  # sentence | token
ap.add_argument("--chunk_tokens", type=int, default=16)
ap.add_argument("--max_sent_tokens", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--query_agnostic", action="store_true")  # baselines only: compress context without question (kvpress default)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)

model, tok = load_model(args.model)
pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tok)
L, H = model.config.num_hidden_layers, model.config.num_attention_heads

ds = load_dataset("simonjegou/ruler", "4096", split="test").to_pandas()
rows = []
for task, g in ds.groupby("task", sort=True):
    if args.tasks != "all" and task not in args.tasks.split(","):
        continue
    ii = list(g.index); random.Random(0).shuffle(ii)
    sel = ii[:args.n_dev] if args.split == "dev" else ii[args.n_dev:args.n_dev + args.n_test]
    rows += sel
df = ds.loc[rows].copy()

heads = None
if args.method == "relhead":
    if args.heads_mode == "all":
        heads = [(l, h) for l in range(L) for h in range(H)]
    elif args.heads_mode == "random":
        rng = random.Random(args.seed); heads = rng.sample([(l, h) for l in range(L) for h in range(H)], args.k)
    else:
        hj = json.load(open(args.heads_file)); A = np.array(hj[f"auroc_{args.agg}"]).ravel()
        order = np.argsort(-A) if args.heads_mode == "top" else np.argsort(A)
        heads = [(int(j // H), int(j % H)) for j in order[:args.k]]
    print("heads:", heads, flush=True)
hbl = heads_to_dict(heads) if heads else None

def make_press():
    r = args.ratio
    return {"full": None, "snapkv": SnapKVPress(compression_ratio=r), "adakv": AdaKVPress(SnapKVPress(compression_ratio=r)),
            "pyramidkv": PyramidKVPress(compression_ratio=r),
            "chunkkv": ChunkKVPress(press=SnapKVPress(compression_ratio=r), chunk_length=20),
            "criticaladakv": CriticalAdaKVPress(press=SnapKVPress(compression_ratio=r))}[args.method]

@torch.no_grad()
def run_relhead(ctx, question, answer_prefix, max_new):
    text, c0, q0, suffix = build_prompt(tok, ctx, question)
    ids, offs = tokenize_with_offsets(tok, text)
    T = len(ids)
    q_start = next(t for t, (a, b) in enumerate(offs) if a >= q0)
    ctx_start = next(t for t, (a, b) in enumerate(offs) if a >= c0)
    cache = DynamicCache()
    with QAttnCapture(model, hbl, q_start) as cap:
        model.model(input_ids=torch.tensor([ids], device=model.device), past_key_values=cache)
    A = torch.cat([cap.out[l] for l in sorted(cap.out)])[:, ctx_start:q_start]  # (k, n_ctx)
    A = A / A.sum(-1, keepdim=True).clamp_min(1e-12)  # each head's distribution over context
    tok_score = torch.zeros(T, device=A.device); tok_score[ctx_start:q_start] = A.mean(0)
    chunks = token_chunks(offs, c0, q0, mode=args.chunk, text=text, chunk_tokens=args.chunk_tokens,
                          max_sent_tokens=args.max_sent_tokens)
    cs = torch.tensor([tok_score[c].sum().item() / (len(c) if args.agg == "density" else 1) for c in chunks])
    forced = list(range(ctx_start)) + list(range(q_start, T))
    n_kept = compute_n_kept(T, args.ratio)
    budget = n_kept - len(forced)
    keep = list(forced)
    for ci in torch.argsort(-cs).tolist():
        if budget <= 0:
            break
        c = chunks[ci]
        if len(c) > budget:  # take the highest-scoring tokens of the last chunk
            c = sorted(c, key=lambda t: -tok_score[t].item())[:budget]
        keep += c; budget -= len(c)
    keep = torch.tensor(sorted(keep))
    prune_cache(cache, keep)
    q_ids = tok.encode(suffix + answer_prefix, add_special_tokens=False, return_tensors="pt").to(model.device)
    ans = pipe.generate_answer(question_ids=q_ids, cache=cache, context_length=T, max_new_tokens=max_new)
    return ans, {"T": T, "kept": len(keep), "n_chunks": len(chunks), "forced": len(forced)}

preds, extras = [], []
t0 = time.time()
for n, (_, r) in enumerate(df.iterrows()):
    max_new = int(r["max_new_tokens"])
    if args.method == "relhead":
        a, ex = run_relhead(r["context"], r["question"], r["answer_prefix"], max_new)
    else:  # kvpress query-aware protocol: question appended to context, question=""
        if args.query_agnostic:
            a = pipe(r["context"], question=r["question"], answer_prefix=r["answer_prefix"], press=make_press(),
                     max_new_tokens=max_new)["answer"]
        else:
            a = pipe(r["context"] + r["question"], question="", answer_prefix=r["answer_prefix"], press=make_press(),
                     max_new_tokens=max_new)["answer"]
        ex = {}
    preds.append(a); extras.append(ex)
    if n % 100 == 0:
        print(f"{n}/{len(df)} {time.time()-t0:.0f}s | {r['task']} pred={a[:60]!r} gold={r['answer']}", flush=True)

df["predicted_answer"] = preds
df["answer"] = df["answer"].apply(lambda s: list(ast.literal_eval(s)) if isinstance(s, str) else list(s))
df["extra"] = [json.dumps(e) for e in extras]
df.drop(columns=["context"]).to_json(args.out.replace(".json", "_preds.jsonl"), orient="records", lines=True)
m = calculate_metrics(df.copy())
avg = float(np.mean([v["string_match"] for v in m.values()]))
res = {"args": vars(args), "heads": heads, "per_task": m, "avg": avg, "n": len(df), "time_s": time.time() - t0}
json.dump(res, open(args.out, "w"), indent=1)
print("AVG", round(avg, 2), {k: v["string_match"] for k, v in m.items()})
