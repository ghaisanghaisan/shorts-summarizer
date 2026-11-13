"""Benchmark Indonesian sentiment models (and Gemini zero-shot) on public + domain data.

usage:
  uv run --extra sentiment python benchmarks/sentiment_bench.py run [--gemini-n 200] [--no-gemini] [--limit N]
  uv run --extra sentiment python benchmarks/sentiment_bench.py export-domain [report.json|out-dir ...]

Datasets
  smsa   IndoNLU SmSA validation (1,260, reviews/comments)         benchmarks/data/smsa_valid.tsv
  nusax  NusaX-senti Indonesian test (400, reviews/everyday text)   benchmarks/data/nusax_ind_test.csv
  domain benchmarks/domain.jsonl  {"text","context","label"}  (hand-labelled short-video snippets)

Contamination: w11wo/* and mdhugol/* were fine-tuned on SmSA. NusaX's Indonesian sentences were
sampled from the same Prosa corpus: 110/400 NusaX test sentences appear verbatim in SmSA train and
319/400 share an 8-word shingle with it. So both smsa and nusax are contaminated for those two models
(marked *). `nusax_clean` keeps only the NusaX test items with no 8-gram overlap with SmSA train:
the only public set here that is fair to every model.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from shorts_summarizer.schema import VideoReport  # noqa: E402
from shorts_summarizer.sentiment import Scorer, build_units, context_of  # noqa: E402

LABELS = ["negative", "neutral", "positive"]
MODELS = {
    "apriandito/indobert-sentiment-classifier": {"contaminated": set()},
    "w11wo/indonesian-roberta-base-sentiment-classifier": {"contaminated": {"smsa", "nusax"}},
    "mdhugol/indonesia-bert-sentiment-classification": {
        "contaminated": {"smsa", "nusax"}, "label_map": {0: "positive", 1: "neutral", 2: "negative"}},
}
# context strategies for the context-conditioned apriandito model
CONTEXTS = {"generic": "Topik umum", "review": "Ulasan produk, tempat atau layanan"}


def load(name: str) -> list[dict]:
    if name == "smsa":
        rows = csv.reader(open(ROOT / "data/smsa_valid.tsv", encoding="utf-8"), delimiter="\t")
        return [{"text": t, "label": l, "context": None} for t, l in rows]
    if name == "nusax":
        return [{"text": r["text"], "label": r["label"], "context": None}
                for r in csv.DictReader(open(ROOT / "data/nusax_ind_test.csv", encoding="utf-8"))]
    if name == "nusax_clean":
        train = {_norm(r[0]) for r in csv.reader(open(ROOT / "data/smsa_train.tsv", encoding="utf-8"), delimiter="\t")}
        train_sh = set().union(*(_shingles(t) for t in train))
        return [it for it in load("nusax") if not (_shingles(_norm(it["text"])) & train_sh)]
    if name == "domain":
        p = ROOT / "domain.jsonl"
        if not p.exists():
            return []
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return [r for r in rows if r.get("label") in LABELS]
    raise ValueError(name)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _shingles(s: str, k: int = 8) -> set[str]:
    w = s.split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def metrics(gold: list[str], pred: list[str]) -> dict:
    return {
        "n": len(gold),
        "acc": round(accuracy_score(gold, pred), 4),
        "macro_f1": round(f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0), 4),
        "f1": dict(zip(LABELS, [round(x, 4) for x in f1_score(gold, pred, labels=LABELS, average=None, zero_division=0)])),
        "confusion": confusion_matrix(gold, pred, labels=LABELS).tolist(),
    }


class Batch(BaseModel):
    labels: list[Literal["negative", "neutral", "positive"]]


def gemini_predict(items: list[dict], batch: int = 25) -> list[str]:
    from dotenv import load_dotenv
    from shorts_summarizer.analyzers.gemini import GeminiAnalyzer

    load_dotenv(ROOT.parent / ".env")
    an = GeminiAnalyzer()
    system = ("You are an Indonesian sentiment classifier. For each numbered text, output its overall sentiment: "
              "negative, neutral or positive. Return exactly one label per text, in order.")
    preds: list[str] = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        body = "\n".join(f"{j + 1}. " + (f"[topik: {it['context']}] " if it.get("context") else "") + it["text"]
                         for j, it in enumerate(chunk))
        out = an._parse(system, [body], Batch).labels
        if len(out) != len(chunk):  # model miscounted: retry one by one
            out = [an._parse(system, [it["text"]], Batch).labels[0] for it in chunk]
        preds += out
    return preds, an.model


def worst(items, probs, preds, k=10):
    errs = [(max(p.values()), it["text"][:120], it["label"], pr) for it, p, pr in zip(items, probs, preds)
            if pr != it["label"]]
    return [{"conf": round(c, 3), "text": t, "gold": g, "pred": p} for c, t, g, p in sorted(errs, reverse=True)[:k]]


def run(args):
    random.seed(13)
    data = {n: load(n) for n in args.datasets}
    data = {n: (d[: args.limit] if args.limit else d) for n, d in data.items() if d}
    subsets = {n: sorted(random.sample(range(len(d)), min(args.gemini_n, len(d)))) for n, d in data.items()}
    print({n: len(d) for n, d in data.items()})
    results = []

    for name, cfg in MODELS.items():
        if args.no_local:
            break
        scorer = Scorer(name, label_map=cfg.get("label_map"))
        strategies = ["generic", "review", "topic"] if scorer.pair else ["none"]
        for ds, items in data.items():
            for strat in strategies:
                if strat == "topic" and ds != "domain":
                    continue
                if strat == "review" and ds == "domain":
                    continue
                ctx = ([it.get("context") or CONTEXTS["generic"] for it in items] if strat == "topic"
                       else CONTEXTS.get(strat))
                t0 = time.perf_counter()
                probs = scorer.probs([it["text"] for it in items], ctx)
                ms = (time.perf_counter() - t0) / len(items) * 1000
                preds = [max(p, key=p.get) for p in probs]
                gold = [it["label"] for it in items]
                sub = subsets[ds]
                results.append({
                    "model": name, "context": strat, "dataset": ds,
                    "contaminated": ds in cfg["contaminated"], "ms_per_item": round(ms, 1),
                    "full": metrics(gold, preds),
                    "subset": metrics([gold[i] for i in sub], [preds[i] for i in sub]),
                    "worst_errors": worst(items, probs, preds),
                })
                r = results[-1]
                print(f"{name:55s} {strat:8s} {ds:7s} acc={r['full']['acc']:.3f} "
                      f"mF1={r['full']['macro_f1']:.3f} {ms:.0f}ms/item", flush=True)
        del scorer

    if not args.no_gemini:
        for ds, items in data.items():
            sub = [items[i] for i in subsets[ds]]
            try:
                preds, gm = gemini_predict(sub)
            except Exception as e:
                print(f"gemini {ds}: {type(e).__name__}: {e}")
                continue
            m = metrics([it["label"] for it in sub], preds)
            results.append({"model": gm, "context": "zero-shot", "dataset": ds, "contaminated": False,
                            "ms_per_item": None, "full": None, "subset": m, "worst_errors": []})
            print(f"{gm:55s} {'0-shot':8s} {ds:7s} acc={m['acc']:.3f} mF1={m['macro_f1']:.3f} (n={m['n']})", flush=True)

    # merge with earlier runs (keyed by model/context/dataset) so partial reruns don't lose results
    rp = ROOT / "results.json"
    old = json.loads(rp.read_text()) if rp.exists() else []
    key = lambda r: (r["model"], r["context"], r["dataset"])
    merged = {key(r): r for r in old} | {key(r): r for r in results}
    for r in merged.values():  # contamination flags follow the current knowledge, not the run date
        r["contaminated"] = r["dataset"] in MODELS.get(r["model"], {}).get("contaminated", set())
    results = list(merged.values())
    rp.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    sizes = {r["dataset"]: (r["full"] or r["subset"])["n"] for r in results}
    write_md(results, sizes, args.gemini_n)


def write_md(results, data, gn):
    L = ["# Sentiment benchmark", "",
         f"Datasets: " + ", ".join(f"{k} (n={v})" for k, v in data.items()) +
         f". Subset = the same {gn} random items per dataset that Gemini saw (fair comparison).",
         "`*` = contaminated: the model was fine-tuned on SmSA, and NusaX's Indonesian text comes from the same corpus.",
         "`nusax_clean` = NusaX test items with no 8-gram overlap with SmSA train (fair to all models).", "",
         "| model | context | dataset | acc | macro-F1 | F1 neg / neu / pos | subset acc | subset macro-F1 | ms/item |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (r["dataset"], -(r["subset"]["macro_f1"]))):
        f, s = r["full"], r["subset"]
        star = "*" if r["contaminated"] else ""
        fl = f or s
        L.append(f"| {r['model'].split('/')[-1]}{star} | {r['context']} | {r['dataset']} | "
                 f"{f['acc']:.3f} | {f['macro_f1']:.3f} | " if f else
                 f"| {r['model']} | {r['context']} | {r['dataset']} | – | – | ")
        L[-1] += (f"{fl['f1']['negative']:.2f} / {fl['f1']['neutral']:.2f} / {fl['f1']['positive']:.2f} | "
                  f"{s['acc']:.3f} | {s['macro_f1']:.3f} | {r['ms_per_item'] if r['ms_per_item'] is not None else '–'} |")
    L += ["", "Confusion matrices (rows = gold neg/neu/pos, cols = predicted) and worst errors: `results.json`."]
    (ROOT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def export_domain(paths: list[str]):
    """Dump unlabelled snippets from reports for hand-labelling into benchmarks/domain.jsonl."""
    reps = []
    for p in paths or [str(ROOT.parent / "out")]:
        p = Path(p)
        reps += [p] if p.is_file() else sorted(p.glob("*/report.json")) or ([p / "report.json"] if (p / "report.json").exists() else [])
    out = ROOT / "domain.jsonl"
    existing = {json.loads(l)["text"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    n = 0
    with out.open("a") as f:
        for rp in reps:
            r = VideoReport.model_validate_json(rp.read_text())
            for u in build_units(r):
                if u.text in existing or len(u.text.split()) < 3:
                    continue
                f.write(json.dumps({"video": r.meta.id, "source": u.source, "t": u.t, "context": context_of(r),
                                    "text": u.text, "label": None}, ensure_ascii=False) + "\n")
                existing.add(u.text)
                n += 1
    print(f"appended {n} unlabelled snippets from {len(reps)} reports -> {out}\n"
          f"set each \"label\" to negative / neutral / positive, then rerun `run`.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--gemini-n", type=int, default=200)
    r.add_argument("--no-gemini", action="store_true")
    r.add_argument("--no-local", action="store_true", help="skip the local HF models")
    r.add_argument("--datasets", nargs="+", default=["smsa", "nusax", "nusax_clean", "domain"])
    r.add_argument("--limit", type=int, default=0, help="debug: cap items per dataset")
    e = sp.add_parser("export-domain")
    e.add_argument("paths", nargs="*")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else export_domain(a.paths)
