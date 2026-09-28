#!/usr/bin/env python3
"""KV Zoo: will this model fit on my GPUs, how long a context can I run, and how fast?

Reads a Hugging Face config.json and works out, layer by layer, how much memory
the KV cache (the "notes" each layer keeps) needs and how much has to be read to
write one new token. Knows full attention, sliding window, DeepSeek sparse
attention (DSA), MLA and linear attention layers.

Standard library only. Estimates, not measurements.

  python3 kv_zoo.py naive --gpu h100 --count 8 --ctx 1m
  python3 kv_zoo.py NaiveAI/Naive-N0.5-Flash --users 4 --kv fp8
  python3 kv_zoo.py ./config.json --total 70 --active 70 --gpu h200 --count 2
"""
import argparse
import json
import math
import os
import sys
import urllib.request

VERSION = "1.0.0"
VIEWER_URL = "https://code415.dev/demos/2026-09-28/kv-zoo"
GB = 1e9
PREC = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int4": 0.5, "4bit": 0.5}

GPUS = {
    "h100": ("NVIDIA H100 80GB", 80, 3350, True),
    "h200": ("NVIDIA H200 141GB", 141, 4800, True),
    "b200": ("NVIDIA B200 180GB", 180, 7700, True),
    "a100": ("NVIDIA A100 80GB", 80, 2039, False),
    "rtx5090": ("GeForce RTX 5090 32GB", 32, 1792, True),
    "rtx4090": ("GeForce RTX 4090 24GB", 24, 1008, True),
    "m3ultra": ("Mac Studio M3 Ultra 512GB", 512, 819, False),
    "spark": ("NVIDIA DGX Spark 128GB", 128, 273, True),
}

# Presets carry the config fields that matter, copied from each repo's config.json,
# plus parameter counts from the model cards. They work offline.
PRESETS = {
    "naive": {
        "name": "Naive-N0.5-Flash", "hf": "NaiveAI/Naive-N0.5-Flash", "total": 309, "active": 15.5, "w": "fp8",
        "config": {"num_hidden_layers": 48, "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 192, "v_head_dim": 128,
                   "hybrid_layer_pattern": [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                   "sliding_window": 128, "swa_num_key_value_heads": 8, "swa_head_dim": 192, "swa_v_head_dim": 128,
                   "enable_dsa": True, "index_top_k": 2048, "index_head_dim": 128, "index_n_kv_heads": 1, "max_position_embeddings": 1048576}},
    "deepseek": {
        "name": "DeepSeek-V3.2", "hf": "deepseek-ai/DeepSeek-V3.2", "total": 671, "active": 37, "w": "fp8",
        "config": {"num_hidden_layers": 61, "num_attention_heads": 128, "num_key_value_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                   "qk_nope_head_dim": 128, "v_head_dim": 128, "index_topk": 2048, "index_head_dim": 128, "max_position_embeddings": 163840}},
    "gptoss": {
        "name": "gpt-oss-120b", "hf": "openai/gpt-oss-120b", "total": 117, "active": 5.1, "w": "int4",
        "config": {"num_hidden_layers": 36, "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64, "sliding_window": 128,
                   "layer_types": ["sliding_attention", "full_attention"] * 18, "max_position_embeddings": 131072}},
    "qnext": {
        "name": "Qwen3-Next-80B-A3B", "hf": "Qwen/Qwen3-Next-80B-A3B-Instruct", "total": 80, "active": 3, "w": "bf16",
        "config": {"num_hidden_layers": 48, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256, "full_attention_interval": 4,
                   "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                   "linear_conv_kernel_dim": 4, "use_sliding_window": False, "max_position_embeddings": 262144}},
    "qwen32": {
        "name": "Qwen3-32B", "hf": "Qwen/Qwen3-32B", "total": 32.8, "active": 32.8, "w": "bf16",
        "config": {"num_hidden_layers": 64, "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128, "sliding_window": None,
                   "use_sliding_window": False, "max_position_embeddings": 40960}},
}

ANIMAL = {"full": "elephant", "swa": "goldfish", "dsa": "elephant + mouse", "mla": "squirrel", "mla_dsa": "squirrel + mouse", "linear": "camel"}
KIND = {"full": "full attention", "swa": "sliding window", "dsa": "sparse attention, DSA", "mla": "MLA", "mla_dsa": "MLA + DSA", "linear": "linear attention"}


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def parse_config(cfg):
    """Turn a config.json dict into a list of layer dicts and a max context."""
    if isinstance(cfg, dict) and "text_config" in cfg and not cfg.get("num_hidden_layers"):
        cfg = dict(cfg["text_config"])
    if not isinstance(cfg, dict):
        raise ValueError("not a config object")
    n = _num(cfg.get("num_hidden_layers")) or _num(cfg.get("n_layer")) or _num(cfg.get("num_layers"))
    if not n:
        raise ValueError("num_hidden_layers is missing")
    n = int(n)
    heads = _num(cfg.get("num_attention_heads")) or _num(cfg.get("n_head"))
    hidden = _num(cfg.get("hidden_size")) or _num(cfg.get("n_embd"))
    kvh = _num(cfg.get("num_key_value_heads")) or heads
    kd = _num(cfg.get("head_dim")) or (hidden / heads if hidden and heads else None)
    vd = _num(cfg.get("v_head_dim")) or kd
    win = _num(cfg.get("sliding_window")) or _num(cfg.get("sliding_window_size"))
    swa = {"kvh": _num(cfg.get("swa_num_key_value_heads")) or kvh, "kd": _num(cfg.get("swa_head_dim")) or kd,
           "vd": _num(cfg.get("swa_v_head_dim")) or vd, "win": win or 4096}
    topk = _num(cfg.get("index_topk")) or _num(cfg.get("index_top_k"))
    dsa_on = bool(topk) and cfg.get("enable_dsa") is not False
    idx = (_num(cfg.get("index_head_dim")) or 128) * (_num(cfg.get("index_n_kv_heads")) or 1) if dsa_on else 0
    mla = (_num(cfg.get("kv_lora_rank")) or 0) + (_num(cfg.get("qk_rope_head_dim")) or 0) if _num(cfg.get("kv_lora_rank")) else 0
    notes = []

    lt = cfg.get("layer_types")
    if isinstance(lt, list) and lt:
        def kind_of(s):
            s = str(s).lower()
            if "slid" in s or "local" in s:
                return "swa"
            if any(w in s for w in ("linear", "mamba", "delta", "ssm", "recurrent")):
                return "linear"
            return "full"
        types = [kind_of(s) for s in lt]
    elif isinstance(cfg.get("hybrid_layer_pattern"), list) and cfg["hybrid_layer_pattern"]:
        types = ["swa" if v else "full" for v in cfg["hybrid_layer_pattern"]]
    elif _num(cfg.get("full_attention_interval")):
        k = int(cfg["full_attention_interval"])
        types = ["full" if (i + 1) % k == 0 else "linear" for i in range(n)]
    elif _num(cfg.get("sliding_window_pattern")) and win:
        k = int(cfg["sliding_window_pattern"])
        types = ["full" if (i + 1) % k == 0 else "swa" for i in range(n)]
    elif win and cfg.get("use_sliding_window") is not False:
        types = ["swa"] * n
        notes.append("sliding_window is set with no layer pattern, so every layer is treated as sliding window")
    else:
        types = ["full"] * n
    if len(types) != n:
        notes.append("layer pattern length %d differs from num_hidden_layers %d; using the pattern" % (len(types), n))

    state = 0
    if "linear" in types:
        vh, kh = _num(cfg.get("linear_num_value_heads")), _num(cfg.get("linear_num_key_heads"))
        lk, lv = _num(cfg.get("linear_key_head_dim")), _num(cfg.get("linear_value_head_dim"))
        ck = _num(cfg.get("linear_conv_kernel_dim")) or 4
        if vh and lk and lv:
            state = vh * lk * lv + ((kh or vh) * lk * 2 + vh * lv) * (ck - 1)
        else:
            notes.append("linear layers found but their state size is unknown; counted as 0")

    layers = []
    for t in types:
        if t == "swa":
            layers.append(dict(k="swa", **swa))
        elif t == "linear":
            layers.append({"k": "linear", "state": state})
        elif mla:
            layers.append({"k": "mla_dsa", "latent": mla, "idx": idx, "topk": topk} if dsa_on else {"k": "mla", "latent": mla})
        elif dsa_on:
            layers.append({"k": "dsa", "kvh": kvh, "kd": kd, "vd": vd, "idx": idx, "topk": topk})
        else:
            layers.append({"k": "full", "kvh": kvh, "kd": kd, "vd": vd})
    if kd is None and any(l["k"] in ("full", "swa", "dsa") for l in layers):
        raise ValueError("head_dim is missing and cannot be derived")
    return layers, _num(cfg.get("max_position_embeddings")), notes


def per_tok(l, p):
    if l["k"] in ("full", "swa"):
        return l["kvh"] * (l["kd"] + l["vd"]) * p
    if l["k"] == "dsa":
        return l["kvh"] * (l["kd"] + l["vd"]) * p + l["idx"]
    if l["k"] == "mla":
        return l["latent"] * p
    if l["k"] == "mla_dsa":
        return l["latent"] * p + l["idx"]
    return 0


def store(l, L, p):
    if l["k"] == "linear":
        return l["state"] * 2  # BF16 state, same size at any length
    kept = min(L, l["win"]) if l["k"] == "swa" else L
    return kept * per_tok(l, p)


def read(l, L, p):
    if l["k"] == "dsa":
        return L * l["idx"] + min(L, l["topk"]) * l["kvh"] * (l["kd"] + l["vd"]) * p
    if l["k"] == "mla_dsa":
        return L * l["idx"] + min(L, l["topk"]) * l["latent"] * p
    return store(l, L, p)


def full_version(layers):
    """The same model with every trick turned off: every layer reads everything."""
    tpl = next((l for l in layers if l["k"] in ("full", "dsa", "swa")), None)
    out = []
    for l in layers:
        if l["k"] in ("swa", "dsa"):
            out.append({"k": "full", "kvh": l["kvh"], "kd": l["kd"], "vd": l["vd"]})
        elif l["k"] == "mla_dsa":
            out.append({"k": "mla", "latent": l["latent"]})
        elif l["k"] == "linear" and tpl:
            out.append({"k": "full", "kvh": tpl["kvh"], "kd": tpl["kd"], "vd": tpl["vd"]})
        else:
            out.append(l)
    return out


def kv_store(layers, L, p):
    return sum(store(l, L, p) for l in layers)


def kv_read(layers, L, p):
    return sum(read(l, L, p) for l in layers)


def plan(m, s):
    name, mem, bw, fp8 = GPUS[s["gpu"]]
    p, wb = PREC[s["kv"]], PREC[s["w"]]
    usable = mem * s["count"] * GB * (1 - s.get("reserve", 0.1))
    weights = (m.get("total") or 0) * 1e9 * wb
    kv_one = kv_store(m["layers"], s["ctx"], p)
    kv_all = kv_one * s["users"]
    free = usable - weights
    max_users = (math.floor(free / kv_one) if kv_one > 0 else float("inf")) if free > 0 else 0
    cap = m.get("max_ctx") or 1048576
    max_ctx = 0
    if free > 0:
        if kv_store(m["layers"], cap, p) * s["users"] <= free:
            max_ctx = cap
        else:
            lo, hi = 0, cap
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if kv_store(m["layers"], mid, p) * s["users"] <= free:
                    lo = mid
                else:
                    hi = mid
            max_ctx = lo
    w_read = (m.get("active") or m.get("total") or 0) * 1e9 * wb
    rd = kv_read(m["layers"], s["ctx"], p)
    total_bw = bw * 1e9 * s["count"]
    return {"gpu": name, "fp8": fp8, "usable": usable, "weights": weights, "kv_one": kv_one, "kv_all": kv_all,
            "fits": weights + kv_all <= usable, "free": free, "max_users": max_users, "max_ctx": max_ctx,
            "w_read": w_read, "kv_read": rd, "tps": total_bw / (w_read + rd) if (w_read + rd) else float("inf"),
            "over_cap": bool(m.get("max_ctx")) and s["ctx"] > m["max_ctx"]}


def advise(m, s, r):
    out = []
    p = PREC[s["kv"]]
    per_gpu = GPUS[s["gpu"]][1] * GB * 0.9
    if r["weights"] > r["usable"]:
        out.append("The weights alone need more than this. Plan on at least %d x %s, or a smaller weight format."
                   % (math.ceil(r["weights"] / per_gpu), r["gpu"]))
    elif not r["fits"]:
        if r["max_ctx"] > 0:
            out.append("The notes do not fit. At this many people the longest context that fits is %s; at this context %d fit."
                       % (fmt_tok(r["max_ctx"]), r["max_users"]))
        else:
            out.append("The weights fit, but there is no room left for notes.")
    if (s["w"] == "fp8" or s["kv"] == "fp8") and not r["fp8"]:
        out.append("%s has no native FP8. Expect FP8 weights or notes to run slower or need conversion." % r["gpu"])
    if s["kv"] in ("bf16", "fp16") and r["kv_all"] / 2 > 0.5 * GB and r["weights"] < r["usable"]:
        users2 = math.floor((r["usable"] - r["weights"]) / (r["kv_one"] / 2))
        out.append("Store notes in FP8 to halve them: frees %s here, and room grows from %s to %d people at this context. Check quality on your own tasks first."
                   % (fmt_b(r["kv_all"] / 2), r["max_users"], users2))
    swa = [l for l in m["layers"] if l["k"] == "swa"]
    if swa:
        small = sum(store(l, s["ctx"], p) for l in swa) * s["users"]
        naive = sum(s["ctx"] * per_tok(l, p) for l in swa) * s["users"]
        if naive - small > 0.5 * GB:
            out.append("%d goldfish layers only need the last %d tokens, %s in total. A server that reserves the full length for every layer would waste %s. Use one that sizes each layer on its own, such as vLLM with its hybrid KV cache manager."
                       % (len(swa), swa[0]["win"], fmt_b(small), fmt_b(naive - small)))
    share = r["kv_read"] / (r["kv_read"] + r["w_read"]) if (r["kv_read"] + r["w_read"]) else 0
    if share > 0.5:
        out.append("Notes are %d%% of what is read per new token, so long prompts slow this model down. Shorter contexts, prefix caching or a sparse attention model help most." % round(share * 100))
    else:
        out.append("Weights are %d%% of what is read per new token, so longer context barely slows it down. Serving more people at once is the cheap win." % round((1 - share) * 100))
    if r["over_cap"]:
        out.append("This model's config allows up to %s tokens; numbers above that are capped." % fmt_tok(m["max_ctx"]))
    return out


def fmt_b(b):
    if b == float("inf"):
        return "inf"
    if b >= GB:
        g = b / GB
        return ("%.0f" if g >= 100 else "%.1f" if g >= 10 else "%.2f") % g + " GB"
    if b >= 1e6:
        m = b / 1e6
        return ("%.0f" if m >= 100 else "%.1f") % m + " MB"
    return "%.0f KB" % max(0, b / 1e3)


def fmt_tok(L):
    if L >= 1048576:
        v = L / 1048576
        return ("%d" % round(v) if abs(v - round(v)) < 0.01 else "%.1f" % v) + "M"
    if L >= 1024:
        return "%dK" % round(L / 1024)
    return str(int(L))


def parse_ctx(s):
    s = str(s).strip().lower()
    mult = 1048576 if s.endswith("m") else 1024 if s.endswith("k") else 1
    return int(round(float(s.rstrip("mk")) * mult))


def load_model(arg, total=None, active=None):
    key = arg.lower()
    for pk, p in PRESETS.items():
        if key in (pk, p["hf"].lower(), p["name"].lower()):
            layers, mx, notes = parse_config(p["config"])
            return {"name": p["name"], "hf": p["hf"], "total": total or p["total"], "active": active or p["active"], "w": p["w"],
                    "layers": layers, "max_ctx": mx, "notes": notes}
    if os.path.exists(arg):
        with open(arg) as f:
            cfg = json.load(f)
        base = os.path.basename(arg)
        name = (os.path.basename(os.path.dirname(os.path.abspath(arg))) if base == "config.json" else os.path.splitext(base)[0]) or "custom"
    elif "/" in arg:
        url = "https://huggingface.co/%s/raw/main/config.json" % arg.strip("/")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "kv-zoo/" + VERSION}), timeout=20) as resp:
                cfg = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            sys.exit("Could not fetch %s: %s\nGated repos need you to download config.json yourself and pass its path." % (url, e))
        name = arg
    else:
        sys.exit("Unknown model %r. Use a preset (%s), a Hugging Face repo like org/name, or a path to config.json." % (arg, ", ".join(PRESETS)))
    layers, mx, notes = parse_config(cfg)
    if not total:
        notes.append("parameter count unknown: pass --total and --active to include the weights")
    return {"name": name, "hf": "", "total": total or 0, "active": active or total or 0, "w": "bf16", "layers": layers, "max_ctx": mx, "notes": notes}


def layer_summary(layers):
    counts = {}
    for l in layers:
        counts[l["k"]] = counts.get(l["k"], 0) + 1
    parts = []
    for k, n in counts.items():
        extra = ""
        l = next(x for x in layers if x["k"] == k)
        if k == "swa":
            extra = ", window %d" % l["win"]
        elif k in ("dsa", "mla_dsa"):
            extra = ", top %d" % l["topk"]
        parts.append("%d %s (%s%s)" % (n, ANIMAL[k], KIND[k], extra))
    return "%d layers: %s" % (len(layers), "; ".join(parts))


def report(m, s, compare=None):
    r = plan(m, s)
    p = PREC[s["kv"]]
    line = "-" * 62
    out = []
    out.append(line)
    out.append("KV ZOO  %s" % m["name"])
    out.append(layer_summary(m["layers"]))
    out.append("Setup: %d x %s, weights %s, notes %s, %d %s, context %s"
               % (s["count"], r["gpu"], s["w"].upper(), s["kv"].upper(), s["users"], "person" if s["users"] == 1 else "people", fmt_tok(s["ctx"])))
    out.append(line)
    rows = [
        ("Fits", ("yes" if r["fits"] else "NO") + ", %s usable" % fmt_b(r["usable"])),
        ("  weights", fmt_b(r["weights"])),
        ("  notes, KV cache", fmt_b(r["kv_all"])),
        ("  free" if r["fits"] else "  over", fmt_b(abs(r["usable"] - r["weights"] - r["kv_all"]))),
        ("Longest context per person", fmt_tok(r["max_ctx"]) if r["max_ctx"] else "none"),
        ("Most people at this context", "no limit from memory" if r["max_users"] == float("inf") else str(max(0, r["max_users"]))),
        ("Read per new token", "%s  (weights %s + notes %s)" % (fmt_b(r["w_read"] + r["kv_read"]), fmt_b(r["w_read"]), fmt_b(r["kv_read"]))),
        ("Speed ceiling, 1 person", "<= %s tok/s  (bandwidth / bytes read; real servers reach a fraction)" % format(int(r["tps"]), ",") if r["tps"] != float("inf") else "n/a"),
    ]
    w = max(len(a) for a, _ in rows) + 2
    out += ["%-*s%s" % (w, a, b) for a, b in rows]
    out.append(line)
    fv = full_version(m["layers"])
    same = all(a == b for a, b in zip(fv, m["layers"]))
    hdr = "%-10s %-14s %-14s" % ("context", "notes/person", "read/token")
    if not same:
        hdr += " %-16s %-14s" % ("if all read all", "read/token")
    if compare:
        hdr += " %s" % compare["name"][:22]
    out.append(hdr)
    wb = PREC[s["w"]]
    for L in [4096, 32768, 131072, 262144, 1048576]:
        if m.get("max_ctx") and L > m["max_ctx"]:
            continue
        wr = (m.get("active") or m.get("total") or 0) * 1e9 * wb
        row = "%-10s %-14s %-14s" % (fmt_tok(L), fmt_b(kv_store(m["layers"], L, p)), fmt_b(wr + kv_read(m["layers"], L, p)))
        if not same:
            row += " %-16s %-14s" % (fmt_b(kv_store(fv, L, p)), fmt_b(wr + kv_read(fv, L, p)))
        if compare:
            cw = (compare.get("active") or compare.get("total") or 0) * 1e9 * wb
            if not compare.get("max_ctx") or L <= compare["max_ctx"]:
                row += " %s notes, %s read" % (fmt_b(kv_store(compare["layers"], L, p)), fmt_b(cw + kv_read(compare["layers"], L, p)))
            else:
                row += " beyond its max"
        out.append(row)
    out.append(line)
    out.append("Ideas")
    for a in advise(m, s, r):
        out.append("  * " + a)
    for n in m.get("notes", []):
        out.append("  ! " + n)
    out.append(line)
    out.append("See it as a 3D zoo: %s" % VIEWER_URL)
    return "\n".join(out), r


def main(argv=None):
    ap = argparse.ArgumentParser(description="Will this model fit, how long a context can I run, and how fast? Estimates from config.json.")
    ap.add_argument("model", nargs="?", default="naive", help="preset (%s), Hugging Face repo org/name, or path to config.json" % ", ".join(PRESETS))
    ap.add_argument("--gpu", default="h100", choices=sorted(GPUS), help="hardware (default h100)")
    ap.add_argument("--count", type=int, default=8, help="how many GPUs (default 8)")
    ap.add_argument("--weights", "-w", default=None, choices=["bf16", "fp16", "fp8", "int4"], help="weight format (default: the model's usual one)")
    ap.add_argument("--kv", default="bf16", choices=["bf16", "fp16", "fp8"], help="KV cache format (default bf16)")
    ap.add_argument("--users", type=int, default=1, help="people served at once (default 1)")
    ap.add_argument("--ctx", default="128k", help="context per person, like 32k or 1m (default 128k)")
    ap.add_argument("--total", type=float, help="total parameters in billions, for models that are not presets")
    ap.add_argument("--active", type=float, help="active parameters per token in billions")
    ap.add_argument("--compare", help="a second model to put in the table")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--version", action="version", version="kv_zoo " + VERSION)
    a = ap.parse_args(argv)
    m = load_model(a.model, a.total, a.active)
    s = {"gpu": a.gpu, "count": a.count, "w": a.weights or m["w"], "kv": a.kv, "users": a.users, "ctx": parse_ctx(a.ctx)}
    cmp_m = load_model(a.compare) if a.compare else None
    text, r = report(m, s, cmp_m)
    if a.json:
        print(json.dumps({"model": m["name"], "setup": s, "layers": m["layers"], "plan": r, "ideas": advise(m, s, r)}, indent=1, default=str))
    else:
        print(text)


if __name__ == "__main__":
    main()
