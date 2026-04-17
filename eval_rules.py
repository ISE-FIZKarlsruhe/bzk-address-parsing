"""
Rule evaluation script for BZK address parsing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from rule_induction import RuleInducer, RuleDeductor, RuleEvaluator

ENTITIES        = ["HouseNumber", "StreetName", "City", "District", "State", "Country"]
METRICS_COLUMNS = ["HouseNumber", "StreetName", "City", "State", "Country"]


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate induced rules for BZK address parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument("--induced-rules", required=True, metavar="JSON",
                   help="induced_rules JSON produced by induce_rules.py")

    # Inference model (needed unless --preds-with/without are both supplied)
    p.add_argument("--inference-model",
                   default="meta-llama/Meta-Llama-3-8B-Instruct",
                   metavar="HF_MODEL_ID")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)

    # Cache: skip re-running inference if predictions already exist
    p.add_argument("--preds-with", default=None, metavar="JSON",
                   help="Pre-computed predictions with rules (skips inference if supplied)")
    p.add_argument("--preds-without", default=None, metavar="JSON",
                   help="Pre-computed predictions without rules (skips inference if supplied)")

    # Data
    p.add_argument("--train-file", default="open_data/bzkopen_addresses_train.csv")
    p.add_argument("--val-file",   default="open_data/bzkopen_addresses_val.csv")
    p.add_argument("--split", default="val", choices=["val", "test"])

    # Few-shot strategy
    p.add_argument("--strategy", default="hybrid",
                   choices=["embedding", "pattern", "hybrid"])
    p.add_argument("--num-examples", type=int, default=3)
    p.add_argument("--pool-size", type=int, default=3)
    p.add_argument("--fallback-threshold", type=float, default=0.92)
    p.add_argument("--embedding-model", default="multi-qa-mpnet-base-dot-v1")
    p.add_argument("--ner-model-dir", default="models/ner_bzk")

    # Inducer settings (must match those used during induction)
    p.add_argument("--hybrid-mode", default="hard",
                   choices=["fixed", "adaptive", "hard"])
    p.add_argument("--distance-threshold", type=float, default=0.3)
    p.add_argument("--coverage-threshold", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--min-similarity", type=float, default=0.0)
    p.add_argument("--min-cluster-eval-size", type=int, default=3)

    # Cluster-level filtering (no extra inference needed)
    p.add_argument("--filter-by-cluster", action="store_true",
                   help="Keep a cluster's rules only if its accuracy_lift > 0 on the eval set. "
                        "No additional inference required — uses Analysis C results.")
    p.add_argument("--min-cluster-accuracy-lift", type=float, default=0.0,
                   help="Minimum cluster-level accuracy lift to keep the cluster's rules (default: 0.0, "
                        "i.e. keep only clusters with lift > 0)")
    p.add_argument("--min-cluster-f1-lift", type=float, default=None,
                   help="Optional additional threshold on cluster-level F1 lift")
    p.add_argument("--cluster-filtered-output", default=None, metavar="JSON",
                   help="Path for cluster-filtered induced rules JSON "
                        "(default: <output-dir>/induced_rules_cluster_filtered.json)")

    # Individual-rule evaluation
    p.add_argument("--run-individual-eval", action="store_true",
                   help="Evaluate each rule individually (re-uses the loaded model)")

    # Rule filtering
    p.add_argument("--filter-rules", action="store_true",
                   help="Write a filtered induced-rules JSON (requires --run-individual-eval)")
    p.add_argument("--min-accuracy-lift", type=float, default=0.0)
    p.add_argument("--min-f1-lift", type=float, default=None)
    p.add_argument("--filtered-rules-output", default=None, metavar="JSON")

    # Output
    p.add_argument("--output-dir", default="exp_results/eval_rules", metavar="DIR")

    return p.parse_args()


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype=str, na_values=[""])


def load_preds(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    preds = data["preds"] if isinstance(data, dict) and "preds" in data else data
    return pd.DataFrame(preds)


def save_preds(preds: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"preds": preds}, f, ensure_ascii=False)


# ── Inference ────────────────────────────────────────────────────────────────

def build_model(args, train_df: pd.DataFrame, induced_rules, inducer, device):
    """Load the model once and return it together with both prompt templates.

    The model weights are loaded a single time.  Callers swap model.prompt
    between the two inference runs rather than loading a second copy.
    """
    from mllms import (
        SimilarExamples, NERPatternSimilarExamples,
        HybridSimilarExamples, FallbackExamplesStrategy,
        LlamaAddressParsingModel, JsonDictPromptTemplate,
        InducedRulesJsonDictPromptTemplate,
    )
    from run import (
        _PROMPT_BASE, _PROMPT_RULES_SUFFIX,
        build_embedding_model, ENTITIES_TO_PREDICT,
    )

    sent_model = build_embedding_model(args, device)

    n = args.num_examples
    if args.strategy == "embedding":
        strategy = SimilarExamples(
            example_addresses=train_df["FullAddress"],
            example_labels=train_df,
            num_examples=n,
            labels_to_include=ENTITIES_TO_PREDICT,
            embedding_model=sent_model,
            device=device,
        )
    elif args.strategy == "pattern":
        strategy = NERPatternSimilarExamples(
            example_addresses=train_df["FullAddress"],
            example_labels=train_df,
            num_examples=n,
            labels_to_include=ENTITIES_TO_PREDICT,
            model_dir=args.ner_model_dir,
        )
    else:  # hybrid
        emb_strat = SimilarExamples(
            example_addresses=train_df["FullAddress"],
            example_labels=train_df,
            num_examples=n,
            labels_to_include=ENTITIES_TO_PREDICT,
            embedding_model=sent_model,
            device=device,
        )
        pat_strat = NERPatternSimilarExamples(
            example_addresses=train_df["FullAddress"],
            example_labels=train_df,
            num_examples=n,
            labels_to_include=ENTITIES_TO_PREDICT,
            model_dir=args.ner_model_dir,
        )
        strategy = HybridSimilarExamples(
            embedding_strategy=emb_strat,
            pattern_strategy=pat_strat,
            num_examples=n,
            pool_size=args.pool_size,
        )

    if args.fallback_threshold > 0:
        strategy = FallbackExamplesStrategy(
            primary=strategy,
            labels_to_include=ENTITIES_TO_PREDICT,
            threshold=args.fallback_threshold,
            num_examples=n,
        )

    entities_str = ", ".join(ENTITIES_TO_PREDICT + ["Other"])
    base_text = _PROMPT_BASE.format(entities=entities_str)
    static_rules_text = base_text.replace(
        "Format the output as a JSON object",
        _PROMPT_RULES_SUFFIX + "Format the output as a JSON object",
    )

    prompt_without = JsonDictPromptTemplate(static_rules_text)

    deductor = RuleDeductor(
        inducer=inducer,
        induced_rules=induced_rules,
        min_similarity=args.min_similarity,
    )
    induced_template = static_rules_text.replace("%(examples)s", "%(rules)s%(examples)s")
    prompt_with = InducedRulesJsonDictPromptTemplate(
        induced_template, rule_deductor=deductor
    )

    # Load model weights exactly once
    model = LlamaAddressParsingModel(
        model_name=args.inference_model,
        prompt=prompt_without,          # starting condition; swapped before second run
        example_strategy=strategy,
        device=device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    return model, prompt_without, prompt_with, deductor


def run_inference(model, addresses: list[str], label: str) -> list[dict]:
    print(f"  Running inference [{label}] on {len(addresses)} addresses…")
    t0 = time.monotonic()
    preds = model.parse_addresses(addresses)
    elapsed = time.monotonic() - t0
    print(f"  Done in {elapsed:.1f}s ({len(addresses)/elapsed:.1f} addr/s)")
    return preds


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _bar_comparison(with_vals, without_vals, labels, title, ylabel, path, figsize=(10, 5)):
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w/2, with_vals,    w, label="With rules",    color="#4C72B0")
    ax.bar(x + w/2, without_vals, w, label="Without rules", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_ylim(0, min(1.05, max(max(with_vals), max(without_vals)) * 1.15))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def _heatmap(data: pd.DataFrame, title: str, path: Path, figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.values.astype(float), aspect="auto", cmap="RdYlGn",
                   vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=9)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.values[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(val) < 0.06 else "white")
    plt.colorbar(im, ax=ax, label="delta (with − without)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def _cluster_lift_scatter(cluster_df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, color in zip(
        axes, ["accuracy_lift", "f1_lift"], ["#4C72B0", "#55A868"]
    ):
        df = cluster_df.sort_values(metric, ascending=False).reset_index(drop=True)
        colors = [color if v >= 0 else "#C44E52" for v in df[metric]]
        ax.bar(range(len(df)), df[metric], color=colors, width=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Cluster (sorted by lift)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Per-cluster {metric.replace('_lift','').upper()} lift")
        ax.tick_params(axis="x", labelbottom=False)
        for rank in range(min(3, len(df))):
            ax.annotate(f"c{int(df.loc[rank,'cluster_id'])}",
                        (rank, df.loc[rank, metric]),
                        textcoords="offset points", xytext=(0, 4),
                        fontsize=7, ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def _importance_bar(df: pd.DataFrame, value_col: str, label_col: str,
                    title: str, xlabel: str, path: Path, top_n: int = 20):
    df = df.head(top_n).copy()
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in df[value_col]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(df[label_col][::-1], df[value_col][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = args.split

    print(f"\n{'='*60}")
    print("BZK Rule Evaluation")
    print(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1] Loading data…")
    train_df = load_csv(args.train_file)
    eval_df  = load_csv(args.val_file if sp == "val"
                        else args.val_file.replace("val", "test"))
    with open(args.induced_rules, encoding="utf-8") as f:
        induced_rules = json.load(f)
    print(f"  train: {len(train_df)}  eval: {len(eval_df)}  "
          f"rule clusters: {len(induced_rules)}")

    # ── Build inducer ────────────────────────────────────────────────────────
    print("\n[2] Building RuleInducer…")
    inducer = RuleInducer(
        addresses=train_df["FullAddress"].fillna("").astype(str),
        labels_df=train_df[ENTITIES].fillna(""),
        labels_to_include=ENTITIES,
        embedding_model=args.embedding_model,
        ner_model_dir=args.ner_model_dir,
        hybrid_mode=args.hybrid_mode,
        alpha=args.alpha,
        coverage_threshold=args.coverage_threshold,
        distance_threshold=args.distance_threshold,
    )

    # ── Run / load predictions ────────────────────────────────────────────────
    preds_with_path    = out_dir / f"preds_with_{sp}.json"
    preds_without_path = out_dir / f"preds_without_{sp}.json"

    cache_with    = Path(args.preds_with)    if args.preds_with    else preds_with_path
    cache_without = Path(args.preds_without) if args.preds_without else preds_without_path

    need_model = not (cache_with.exists() and cache_without.exists())

    if need_model:
        print("\n[3] Loading inference model (once)…")
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")
        model, prompt_without, prompt_with, _ = build_model(
            args, train_df, induced_rules, inducer, device
        )
        addresses = eval_df["FullAddress"].fillna("").tolist()

        if not cache_without.exists():
            print("\n[3a] Inference — WITHOUT induced rules…")
            model.prompt = prompt_without
            preds_without = run_inference(model, addresses, "without rules")
            save_preds(preds_without, cache_without)
            print(f"  Cached → {cache_without}")
        else:
            print(f"\n[3a] Loading cached preds_without from {cache_without}")
            preds_without = load_preds(str(cache_without)).to_dict("records")

        if not cache_with.exists():
            print("\n[3b] Inference — WITH induced rules…")
            model.prompt = prompt_with
            preds_with = run_inference(model, addresses, "with rules")
            save_preds(preds_with, cache_with)
            print(f"  Cached → {cache_with}")
        else:
            print(f"\n[3b] Loading cached preds_with from {cache_with}")
            preds_with = load_preds(str(cache_with)).to_dict("records")
    else:
        print(f"\n[3] Loading cached predictions…")
        print(f"  with    : {cache_with}")
        print(f"  without : {cache_without}")
        preds_with    = load_preds(str(cache_with)).to_dict("records")
        preds_without = load_preds(str(cache_without)).to_dict("records")

    preds_with_df    = pd.DataFrame(preds_with)
    preds_without_df = pd.DataFrame(preds_without)

    # ── Build evaluator ───────────────────────────────────────────────────────
    print("\n[4] Building RuleEvaluator…")
    evaluator = RuleEvaluator(
        inducer=inducer,
        induced_rules=induced_rules,
        addresses=eval_df["FullAddress"].fillna("").astype(str),
        ground_truth=eval_df[ENTITIES],
        labels_to_include=ENTITIES,
        min_similarity=args.min_similarity,
    )

    # ── Analysis A: overall A/B ───────────────────────────────────────────────
    print("\n[5] Running analyses…")
    print("  (A) Overall A/B…")
    ab_df = evaluator.ab_test(preds_with_df, preds_without_df, columns=METRICS_COLUMNS)
    ab_df.to_csv(out_dir / f"ab_overall_{sp}.csv")
    core = ["accuracy", "precision", "recall", "f1"]
    _bar_comparison(
        with_vals    = [ab_df.loc[m, "with_rules"]    for m in core],
        without_vals = [ab_df.loc[m, "without_rules"] for m in core],
        labels=core,
        title=f"Overall A/B — with vs without induced rules ({sp})",
        ylabel="Score",
        path=out_dir / f"plot_ab_overall_{sp}.png",
    )

    # ── Analysis B: per-label A/B ─────────────────────────────────────────────
    print("  (B) Per-label A/B…")
    label_df = evaluator.per_label_ab_test(preds_with_df, preds_without_df)
    label_df.to_csv(out_dir / f"ab_per_label_{sp}.csv")
    for metric in ("accuracy", "f1"):
        _bar_comparison(
            with_vals    = list(label_df[f"{metric}_with"]),
            without_vals = list(label_df[f"{metric}_without"]),
            labels=list(label_df.index),
            title=f"Per-label {metric.upper()} — with vs without rules ({sp})",
            ylabel=metric.upper(),
            path=out_dir / f"plot_ab_per_label_{metric}_{sp}.png",
        )
    delta_cols = [c for c in label_df.columns if c.endswith("_delta")]
    delta_df = label_df[delta_cols].copy()
    delta_df.columns = [c.replace("_delta", "") for c in delta_df.columns]
    _heatmap(delta_df.T,
             title=f"Per-label delta (with − without rules) [{sp}]",
             path=out_dir / f"plot_ab_delta_heatmap_{sp}.png")

    # ── Analysis C: per-cluster ───────────────────────────────────────────────
    print("  (C) Per-cluster metrics…")
    cluster_df = evaluator.per_cluster_metrics(
        preds_with_df, preds_without_df,
        min_cluster_eval_size=args.min_cluster_eval_size,
    )
    cluster_df.to_csv(out_dir / f"per_cluster_{sp}.csv", index=False)
    if not cluster_df.empty:
        _cluster_lift_scatter(cluster_df, out_dir / f"plot_cluster_lift_{sp}.png")

    # ── Cluster-level filtering (optional, no extra inference) ───────────────
    if args.filter_by_cluster and not cluster_df.empty:
        print("\n[C+] Filtering clusters by accuracy lift…")
        keep_mask = cluster_df["accuracy_lift"] > args.min_cluster_accuracy_lift
        if args.min_cluster_f1_lift is not None:
            keep_mask &= cluster_df["f1_lift"] > args.min_cluster_f1_lift
        kept_ids   = set(cluster_df.loc[keep_mask, "cluster_id"].astype(int).tolist())
        dropped_ids = set(cluster_df["cluster_id"].astype(int).tolist()) - kept_ids
        print(f"  Keeping {len(kept_ids)} clusters, dropping {len(dropped_ids)}: {sorted(dropped_ids)}")
        # Keep ALL clusters in the JSON but zero out rules_text for dropped ones.
        # This prevents addresses from re-matching to the wrong surviving cluster —
        # they still match their original cluster, but get empty (no) rules.
        cluster_filtered = []
        for r in induced_rules:
            entry = dict(r)
            if int(r["cluster_id"]) in dropped_ids:
                entry["rules_text"] = ""
            cluster_filtered.append(entry)
        cf_path = Path(
            args.cluster_filtered_output
            or out_dir / "induced_rules_cluster_filtered.json"
        )
        cf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cf_path, "w", encoding="utf-8") as f:
            json.dump(cluster_filtered, f, ensure_ascii=False, indent=2)
        print(f"  Saved cluster-filtered rules ({len(cluster_filtered)} clusters) → {cf_path}")
        print(f"  Re-run with: --induced-rules {cf_path} --output-dir <new_dir>")

    # ── Analysis D: rule importance ───────────────────────────────────────────
    print("  (D) Rule importance…")
    imp_df = evaluator.cluster_importance(
        preds_with_df, preds_without_df,
        min_cluster_eval_size=args.min_cluster_eval_size,
    )
    imp_df.to_csv(out_dir / f"rule_importance_{sp}.csv")
    if not imp_df.empty:
        imp_df["label"] = imp_df["cluster_id"].apply(lambda c: f"cluster {int(c)}")
        _importance_bar(
            imp_df, value_col="weighted_contribution", label_col="label",
            title=f"Cluster-level rule importance ({sp})",
            xlabel="Weighted contribution (accuracy_lift × eval_share)",
            path=out_dir / f"plot_rule_importance_{sp}.png",
        )

    # ── Analysis E (optional): individual rule evaluation ─────────────────────
    ind_df = None
    if args.run_individual_eval:
        print("\n[6] Individual rule evaluation…")

        # Reuse the already-loaded model if inference was run above,
        # otherwise load it now (once).
        if not need_model:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, prompt_without, prompt_with, _ = build_model(
                args, train_df, induced_rules, inducer, device
            )

        # Stub deductor: always returns a fixed rules_text for any address.
        # We swap its text before each parse_fn call so we reuse the model.
        class _FixedDeductor:
            def __init__(self): self._text = ""
            def get_rules(self, _): return self._text

        stub = _FixedDeductor()
        model.prompt = prompt_with
        model.prompt.rule_deductor = stub

        def parse_fn(addresses: list[str], rules_text: str) -> list[dict]:
            stub._text = rules_text
            return model.parse_addresses(addresses)

        ind_df = evaluator.evaluate_rules_individually(
            parse_fn=parse_fn,
            preds_with=preds_with_df,
            min_cluster_eval_size=args.min_cluster_eval_size,
        )
        ind_df.to_csv(out_dir / f"individual_rules_{sp}.csv", index=False)

        if not ind_df.empty:
            ind_df["label"] = ind_df.apply(
                lambda r: f"c{int(r['cluster_id'])} r{int(r['rule_rank'])}", axis=1
            )
            _importance_bar(
                ind_df.sort_values("accuracy_lift", ascending=False),
                value_col="accuracy_lift", label_col="label",
                title=f"Individual rule accuracy lift — LOO ({sp})",
                xlabel="Accuracy lift (all rules) − (all rules except this one)",
                path=out_dir / f"plot_individual_rule_lift_{sp}.png",
            )
            # Scatter: accuracy_lift vs f1_lift
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in ind_df["accuracy_lift"]]
            ax.scatter(ind_df["accuracy_lift"], ind_df["f1_lift"], c=colors, alpha=0.7, s=40)
            ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
            ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")
            ax.set_xlabel("Accuracy lift")
            ax.set_ylabel("Macro-F1 lift")
            ax.set_title(f"Individual rule lift — LOO ({sp})")
            fig.tight_layout()
            fig.savefig(out_dir / f"plot_individual_rule_scatter_{sp}.png", dpi=150)
            plt.close(fig)
            print(f"  Saved → {out_dir}/plot_individual_rule_scatter_{sp}.png")

        # ── Rule filtering ────────────────────────────────────────────────────
        if args.filter_rules:
            print("\n[7] Filtering rules…")
            filtered_rules, eval_kept = evaluator.filter_rules(
                parse_fn=parse_fn,
                preds_with=preds_with_df,
                min_accuracy_lift=args.min_accuracy_lift,
                min_f1_lift=args.min_f1_lift,
                min_cluster_eval_size=args.min_cluster_eval_size,
            )
            default_out = out_dir / "induced_rules_filtered.json"
            filtered_path = Path(args.filtered_rules_output or default_out)
            filtered_path.parent.mkdir(parents=True, exist_ok=True)
            with open(filtered_path, "w", encoding="utf-8") as f:
                json.dump(filtered_rules, f, ensure_ascii=False, indent=2)
            eval_kept.to_csv(out_dir / f"individual_rules_kept_{sp}.csv", index=False)
            print(f"  Filtered rules → {filtered_path}")
            print(f"  Kept {int(eval_kept['kept'].sum())} / {len(eval_kept)} rules "
                  f"(lift >= {args.min_accuracy_lift})")
    else:
        print("\n[6] Individual rule evaluation skipped "
              "(pass --run-individual-eval to enable).")

    # ── Console summary + text report ─────────────────────────────────────────
    evaluator.summary(preds_with_df, preds_without_df)

    lines = [
        f"BZK Rule Evaluation Report  [{sp}]",
        f"induced_rules : {args.induced_rules}",
        f"preds cached  : {cache_with} / {cache_without}",
        "",
        "── Overall A/B ──",
        ab_df.to_string(),
        "",
        "── Per-label A/B ──",
        label_df.to_string(),
        "",
        "── Per-cluster (top 30 by accuracy lift) ──",
        cluster_df.head(30).to_string(index=False) if not cluster_df.empty else "(none)",
        "",
        "── Rule importance (top 20) ──",
        imp_df.head(20).drop(columns=["rules_text", "label"], errors="ignore").to_string()
        if not imp_df.empty else "(none)",
    ]
    if ind_df is not None and not ind_df.empty:
        lines += [
            "",
            "── Individual rules (top 20 by accuracy lift) ──",
            ind_df.head(20).drop(columns=["per_label_accuracy", "label"],
                                 errors="ignore").to_string(index=False),
        ]

    report_path = out_dir / f"report_{sp}.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved report   → {report_path}")
    print(f"All outputs in → {out_dir}/\n")


if __name__ == "__main__":
    main()
