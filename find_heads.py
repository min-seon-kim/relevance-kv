"""Offline relevance-head discovery on HotpotQA (distractor). For every attention head, average
question-token attention to each sentence; per-question AUROC of evidence (supporting facts) vs other
sentences; average over questions. Saves AUROC for 'mass' (sum over sentence tokens) and 'density'
(mean per token) aggregations."""
import argparse, json, random, time
import numpy as np, torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from common import load_model, build_prompt, tokenize_with_offsets, QAttnCapture

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama3.1-8b")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--offset", type=int, default=0, help="start index into shuffled train set (0=discovery, 100=held-out)")
ap.add_argument("--out", required=True)
args = ap.parse_args()

model, tok = load_model(args.model)
L, H = model.config.num_hidden_layers, model.config.num_attention_heads
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
idx = list(range(len(ds))); random.Random(0).shuffle(idx)
idx = idx[args.offset:args.offset + args.n]

auc = {"mass": [], "density": []}
t0 = time.time()
for n_done, i in enumerate(idx):
    ex = ds[i]
    sup = set(zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]))
    ctx, spans, labels = "Answer the question based on the following passages.\n\n", [], []
    for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
        ctx += f"{title}\n"
        for sid, s in enumerate(sents):
            a = len(ctx); ctx += s; spans.append((a, len(ctx))); labels.append(int((title, sid) in sup))
        ctx += "\n\n"
    question = f"Question: {ex['question']}"
    text, c0, q0, _ = build_prompt(tok, ctx, question)
    ids, offs = tokenize_with_offsets(tok, text)
    q_start = next(t for t, (a, b) in enumerate(offs) if a >= q0)
    sent_tok = []
    for (a, b) in spans:
        tt = [t for t, (x, y) in enumerate(offs) if x >= c0 + a and x < c0 + b and y > x]
        sent_tok.append(tt)
    keep = [k for k, tt in enumerate(sent_tok) if tt]
    lab = np.array([labels[k] for k in keep])
    if lab.sum() == 0 or lab.sum() == len(lab):
        continue
    with torch.no_grad(), QAttnCapture(model, {l: list(range(H)) for l in range(L)}, q_start) as cap:
        model.model(input_ids=torch.tensor([ids], device=model.device), past_key_values=__import__("transformers").DynamicCache())
    A = torch.stack([cap.out[l] for l in range(L)])  # (L, H, k_len)
    seg = torch.zeros(len(keep), A.shape[-1], device=A.device)
    for r, k in enumerate(keep):
        seg[r, sent_tok[k]] = 1.0
    mass = torch.einsum("lhk,sk->lhs", A, seg).cpu().numpy()
    dens = mass / seg.sum(1).cpu().numpy()[None, None]
    for name, M in (("mass", mass), ("density", dens)):
        flat = M.reshape(L * H, -1)
        auc[name].append([roc_auc_score(lab, flat[j]) for j in range(L * H)])
    if n_done % 10 == 0:
        print(f"{n_done}/{len(idx)} ctx_tokens={len(ids)} {time.time()-t0:.0f}s", flush=True)

res = {"model": args.model, "n_questions": len(auc["mass"]), "L": L, "H": H, "offset": args.offset}
for name in auc:
    m = np.array(auc[name]).mean(0)
    res[f"auroc_{name}"] = m.reshape(L, H).tolist()
    order = np.argsort(-m)[:20]
    res[f"top_{name}"] = [[int(j // H), int(j % H), float(m[j])] for j in order]
    print(name, "top10:", [(int(j // H), int(j % H), round(float(m[j]), 3)) for j in order[:10]],
          "median:", round(float(np.median(m)), 3))
json.dump(res, open(args.out, "w"))
