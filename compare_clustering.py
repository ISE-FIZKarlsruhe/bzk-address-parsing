"""
Compare soft-adaptive vs hard-selection clustering across coverage thresholds.

Usage:
  uv run python compare_clustering.py
  uv run python compare_clustering.py --distance-threshold 0.25 --train-file open_data/bzkopen_addresses_train.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sentence_transformers
import spacy
from sklearn.metrics import silhouette_score

from rule_induction import (
    _embedding_similarity_matrix,
    _pattern_similarity_matrix,
    _ner_coverage,
    cluster_by_similarity,
    ner_address_to_pattern,
    _BZK_LABELS,
)

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


# CLI 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file",         default="open_data/bzkopen_addresses_train.csv")
    p.add_argument("--embedding-model",    default="multi-qa-mpnet-base-dot-v1")
    p.add_argument("--ner-model-dir",      default="models/ner_bzk")
    p.add_argument("--distance-threshold", type=float, default=0.3)
    p.add_argument("--out-dir",            default="clustering")
    return p.parse_args()


# Helpers 

def _cluster_stats(labels: np.ndarray) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    return {
        "n_clusters":  len(unique),
        "size_min":    int(counts.min()),
        "size_max":    int(counts.max()),
        "size_mean":   float(counts.mean()),
        "singletons":  int((counts == 1).sum()),
    }


def _silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score from a precomputed distance matrix.
    Returns NaN if there is only one cluster."""
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        return float("nan")
    return float(silhouette_score(dist, labels, metric="precomputed"))


def _plot_clusters(
    dist: np.ndarray,
    labels: np.ndarray,
    patterns: list[str],
    title: str,
    path: Path,
    max_labels: int = 30,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    kw = dict(
        n_components=2, metric="precomputed",
        init="random", random_state=42,
        perplexity=min(30, len(dist) - 1),
    )
    coords = TSNE(**kw).fit_transform(dist)

    unique_ids = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", min(len(unique_ids), 20))
    colors = [cmap(int(c) % 20) for c in labels]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=18, alpha=0.7, linewidths=0)

    # Annotate centroid of largest clusters
    sizes = {cid: int((labels == cid).sum()) for cid in unique_ids}
    top = sorted(unique_ids, key=lambda c: sizes[c], reverse=True)[:max_labels]
    for cid in top:
        idx = np.where(labels == cid)[0]
        # Most central = highest avg similarity within cluster
        sub_dist = dist[np.ix_(idx, idx)]
        central = idx[np.argmin(sub_dist.mean(axis=1))]
        cx, cy = coords[central, 0], coords[central, 1]
        lbl = patterns[central]
        if len(lbl) > 35:
            lbl = lbl[:33] + "…"
        ax.annotate(
            f"[{sizes[cid]}] {lbl}", xy=(cx, cy),
            fontsize=6.5, xytext=(4, 4), textcoords="offset points",
        )

    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {path}")


def _save_csv(
    addresses: list[str],
    patterns: list[str],
    labels: np.ndarray,
    coverage: np.ndarray,
    path: Path,
):
    sizes = {cid: int((labels == cid).sum()) for cid in np.unique(labels)}
    df = pd.DataFrame({
        "address":      addresses,
        "pattern":      patterns,
        "cluster_id":   labels,
        "cluster_size": [sizes[c] for c in labels],
        "coverage":     coverage,
    })
    df.to_csv(path, index=True, index_label="idx")
    print(f"  CSV  saved → {path}")


# Main 

def main():
    args = parse_args()
    out_root = Path(args.out_dir)

    # Load data 
    print("Loading data…")
    train_df  = pd.read_csv(args.train_file)
    addresses = train_df["FullAddress"].fillna("").astype(str).tolist()

    # Load models 
    print(f"Loading embedding model '{args.embedding_model}'…")
    emb_model = sentence_transformers.SentenceTransformer(args.embedding_model)

    print(f"Loading NER model from '{args.ner_model_dir}'…")
    nlp = spacy.load(args.ner_model_dir)

    # Compute shared prerequisites (done once) 
    print(f"Computing NER patterns for {len(addresses)} addresses…")
    patterns = [ner_address_to_pattern(a, nlp) for a in addresses]

    print("Computing embedding similarity matrix…")
    emb_sim = _embedding_similarity_matrix(addresses, emb_model)

    print("Computing pattern similarity matrix…")
    pat_sim = _pattern_similarity_matrix(patterns)

    print("Computing NER coverage per address…")
    coverage = np.array([_ner_coverage(a, nlp) for a in addresses], dtype=np.float32)

    # Coverage distribution info 
    print(f"\nCoverage distribution:")
    print(f"  mean={coverage.mean():.3f}  median={np.median(coverage):.3f}"
          f"  min={coverage.min():.3f}  max={coverage.max():.3f}")
    for t in THRESHOLDS:
        frac = (coverage > t).mean()
        print(f"  > {t:.1f}: {frac*100:.1f}% of addresses use pattern")

    # Build configs
    # Each entry: (name, sim_matrix, coverage_array)
    configs = []

    # Soft adaptive baseline
    emb_weights = 1.0 - coverage
    w = (emb_weights[:, None] + emb_weights[None, :]) / 2.0
    sim_adaptive = np.clip(w * emb_sim + (1.0 - w) * pat_sim, 0.0, 1.0)
    configs.append(("soft_adaptive", sim_adaptive, coverage))

    # Hard selection at each threshold
    for t in THRESHOLDS:
        sel_pat  = coverage > t
        both_pat = sel_pat[:, None] & sel_pat[None, :]
        both_emb = (~sel_pat)[:, None] & (~sel_pat)[None, :]
        sim = np.where(both_pat, pat_sim,
              np.where(both_emb, emb_sim,
                       (emb_sim + pat_sim) / 2.0))
        sim = np.clip(sim, 0.0, 1.0)
        configs.append((f"hard_t{int(t*10):02d}", sim, coverage))

    # Run experiments 
    summary_rows = []

    for name, sim, cov in configs:
        print(f"\n── {name} ──")
        run_dir = out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        dist   = np.clip(1.0 - sim, 0.0, 1.0)
        labels = cluster_by_similarity(sim, distance_threshold=args.distance_threshold)
        stats  = _cluster_stats(labels)
        sil    = _silhouette(dist, labels)

        print(f"  clusters={stats['n_clusters']}  "
              f"singletons={stats['singletons']}  "
              f"size min/mean/max={stats['size_min']}/{stats['size_mean']:.1f}/{stats['size_max']}  "
              f"silhouette={sil:.4f}")

        _save_csv(addresses, patterns, labels, cov, run_dir / "clusters.csv")
        _plot_clusters(
            dist, labels, addresses, patterns,
            title=f"{name}  |  k={stats['n_clusters']}  sil={sil:.3f}  (dist_thr={args.distance_threshold})",
            path=run_dir / "clusters.png",
        )

        summary_rows.append({
            "config":        name,
            "n_clusters":    stats["n_clusters"],
            "singletons":    stats["singletons"],
            "size_min":      stats["size_min"],
            "size_mean":     round(stats["size_mean"], 1),
            "size_max":      stats["size_max"],
            "silhouette":    round(sil, 4),
        })

    # Summary table 
    summary = pd.DataFrame(summary_rows)
    summary_path = out_root / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n\n═══ Summary ═══")
    print(summary.to_string(index=False))
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
