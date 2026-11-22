"""Shared utilities: model loading, per-head question->context attention capture,
chunking, cache pruning. Mirrors kvpress tokenization/generation so methods are comparable."""
import math, re, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
from kvpress.utils import get_prerope_query_states

MODELS = {
    "llama3.1-8b": "unsloth/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}


def load_model(name, attn="flash_attention_2"):
    path = MODELS.get(name, name)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attn,
                                                 device_map="cuda:0").eval()
    return model, tok


def build_prompt(tok, context, question_text):
    """Query-aware layout identical to kvpress (query_aware=True):
    compressed part = chat-prefix + context + question ; generation part = chat-suffix + answer_prefix.
    Returns full compressed string, char offset where context starts, where question starts, and suffix."""
    body = context + question_text
    sep = "#" * (len(body) + 10)
    full_t = tok.apply_chat_template([{"role": "user", "content": body + sep}], add_generation_prompt=True, tokenize=False)
    text, suffix = full_t.split(sep)  # exactly kvpress preprocess (template may trim leading spaces)
    pre_t = tok.apply_chat_template([{"role": "user", "content": sep}], add_generation_prompt=True, tokenize=False)
    prefix = pre_t.split(sep)[0]
    assert text.startswith(prefix)
    shift = len(body) - (len(text) - len(prefix))  # leading chars removed by template
    assert body[shift:] == text[len(prefix):]
    return text, len(prefix), len(prefix) + len(context) - shift, suffix


def tokenize_with_offsets(tok, text):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


class QAttnCapture:
    """Forward hooks on chosen layers. After the layer's prefill forward, compute the attention
    distribution of query positions [q_start:] over all keys (causal) for the chosen heads, averaged
    over query positions. Result: self.out[layer] -> (n_heads_sel, k_len) float32."""

    def __init__(self, model, heads_by_layer, q_start):
        self.model, self.hbl, self.q_start = model, heads_by_layer, q_start
        self.out, self.handles = {}, []

    def __enter__(self):
        for li, heads in self.hbl.items():
            attn = self.model.model.layers[li].self_attn
            self.handles.append(attn.register_forward_hook(self._make_hook(li, heads), with_kwargs=True))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()

    def _make_hook(self, li, heads):
        heads_t = torch.tensor(heads, device=self.model.device)

        @torch.no_grad()
        def hook(module, args, kwargs, output):
            hs = kwargs["hidden_states"]
            cos, sin = kwargs["position_embeddings"]
            cache = kwargs["past_key_values"]
            keys = cache.layers[li].keys  # (1, kvh, k_len, d) post-RoPE
            qs = self.q_start
            q = get_prerope_query_states(module, hs[:, qs:])  # (1, H, nq, d)
            q = q * cos[:, qs:].unsqueeze(1) + rotate_half(q) * sin[:, qs:].unsqueeze(1)
            q = q[:, heads_t]
            groups = module.config.num_attention_heads // module.config.num_key_value_heads
            k = keys[:, heads_t // groups]  # map query head -> kv head
            k_len = k.shape[2]
            w = torch.matmul(q.float(), k.float().transpose(2, 3)) / math.sqrt(module.head_dim)
            nq = q.shape[2]
            # causal: query at absolute pos qs+i sees keys <= qs+i
            mask = torch.arange(k_len, device=w.device)[None, :] > (qs + torch.arange(nq, device=w.device))[:, None]
            w = w.masked_fill(mask, float("-inf")).softmax(-1)
            self.out[li] = w.mean(2)[0]  # (n_sel, k_len)
        return hook


def heads_to_dict(heads):
    d = {}
    for l, h in heads:
        d.setdefault(int(l), []).append(int(h))
    return d


def sentence_spans(text, start, end, max_chars=None):
    """Split text[start:end] into sentence-like char spans (newline or .!? followed by space)."""
    spans, s = [], start
    for m in re.finditer(r"(?<=[.!?])\s+|\n+", text[start:end]):
        e = start + m.end()
        if e > s:
            spans.append((s, e))
        s = e
    if s < end:
        spans.append((s, end))
    return spans


def token_chunks(offsets, c0, c1, mode="sentence", text=None, chunk_tokens=16, max_sent_tokens=64):
    """Return list of token-index lists for the context region [c0, c1) in char space."""
    tok_idx = [i for i, (a, b) in enumerate(offsets) if a >= c0 and a < c1 and b > a]
    if mode == "token":
        return [tok_idx[i:i + chunk_tokens] for i in range(0, len(tok_idx), chunk_tokens)]
    spans = sentence_spans(text, c0, c1)
    chunks, j = [], 0
    for (a, b) in spans:
        cur = []
        while j < len(tok_idx) and offsets[tok_idx[j]][0] < b:
            cur.append(tok_idx[j]); j += 1
        if not cur:
            continue
        for i in range(0, len(cur), max_sent_tokens):
            chunks.append(cur[i:i + max_sent_tokens])
    if j < len(tok_idx):
        chunks.append(tok_idx[j:])
    return chunks


def prune_cache(cache, keep_idx):
    idx = keep_idx.to(cache.layers[0].keys.device)
    for layer in cache.layers:
        layer.keys = layer.keys.index_select(2, idx).contiguous()
        layer.values = layer.values.index_select(2, idx).contiguous()
