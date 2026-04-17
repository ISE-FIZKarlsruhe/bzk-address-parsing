"""

# Induce rules from training set with default settings:
  uv run python induce_rules.py

# More fine-grained clusters, pure pattern similarity:
  uv run python induce_rules.py --distance-threshold 0.25 --alpha 0.0

# Fixed number of clusters:
  uv run python induce_rules.py --n-clusters 30 \\
      --embedding-model hm-haitham/xlm-roberta-large-address-parser
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import transformers

from rule_induction import RuleInducer

ENTITIES = ["HouseNumber", "StreetName", "City", "District", "State", "Country"]


def parse_args():
    p = argparse.ArgumentParser(description="Induce address-parsing rules via LLM clustering")

    # Data
    p.add_argument("--train-file", default="open_data/bzkopen_addresses_train.csv")

    # Clustering
    p.add_argument("--hybrid-mode", default="hard", choices=["fixed", "adaptive", "hard"],
                   help="'fixed': global alpha blend; "
                        "'adaptive': soft per-address weight from NER coverage; "
                        "'hard': hard-selection per-address (default)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Embedding weight in fixed hybrid mode (ignored in adaptive/hard mode)")
    p.add_argument("--coverage-threshold", type=float, default=0.5,
                   help="NER coverage threshold for hard mode: above → pattern, below → embedding (default: 0.5)")
    p.add_argument("--distance-threshold", type=float, default=0.3,
                   help="Agglomerative clustering distance threshold (1 − min_similarity)")
    p.add_argument("--n-clusters", type=int, default=None,
                   help="Fixed number of clusters (overrides --distance-threshold)")
    p.add_argument("--embedding-model", default="multi-qa-mpnet-base-dot-v1")
    p.add_argument("--ner-model-dir", default="models/ner_bzk")
    p.add_argument("--max-cluster-examples", type=int, default=8,
                   help="Max examples shown per cluster in the induction prompt")
    p.add_argument("--min-cluster-size", type=int, default=2,
                   help="Skip clusters smaller than this")

    # LLM
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="Model for rule induction. Use a HuggingFace model ID for local "
                        "inference, or an OpenAI model name (e.g. gpt-4.1-mini, gpt-4o) "
                        "to use the OpenAI API (requires OPENAI_API in .env).")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max tokens for local HuggingFace models (ignored for OpenAI)")
    p.add_argument("--max-rules", type=int, default=5,
                   help="Rules to induce per cluster")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Clusters processed per LLM batch (ignored for OpenAI)")

    # Output
    p.add_argument("--output", default=None,
                   help="Path to write the induced rules JSON. "
                        "Defaults to induced_rules_<model>_<timestamp>.json")
    p.add_argument("--disable-thinking", action="store_true",
                   help="Pass enable_thinking=False to the chat template (use with Qwen3 and other reasoning models)")
    p.add_argument("--summary-only", action="store_true",
                   help="Print cluster summary and exit without running the LLM")

    return p.parse_args()


def main():
    args = parse_args()

    # Load data 
    train_df = pd.read_csv(args.train_file)
    addresses = train_df["FullAddress"].fillna("").astype(str)
    labels_df = train_df[ENTITIES].fillna("")

    # Build inducer 
    inducer = RuleInducer(
        addresses=addresses,
        labels_df=labels_df,
        labels_to_include=ENTITIES,
        embedding_model=args.embedding_model,
        ner_model_dir=args.ner_model_dir,
        hybrid_mode=args.hybrid_mode,
        alpha=args.alpha,
        coverage_threshold=args.coverage_threshold,
        distance_threshold=args.distance_threshold,
        n_clusters=args.n_clusters,
        max_cluster_examples=args.max_cluster_examples,
        max_rules_per_cluster=args.max_rules,
        min_cluster_size=args.min_cluster_size,
    )

    summary = inducer.cluster_summary()
    print("\nCluster summary (top 20):")
    print(summary.head(20).to_string(index=False))

    if args.summary_only:
        return

    # OpenAI models
    _OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "text-")
    is_openai = any(args.model.startswith(p) for p in _OPENAI_PREFIXES)

    if is_openai:
        print(f"\nUsing OpenAI model '{args.model}'…")
        results = inducer.induce_rules_openai(
            model=args.model,
            max_tokens=args.max_new_tokens,
        )
    else:
        # Load local HuggingFace model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading LLM '{args.model}' on {device}…")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model, padding_side="left"
        )
        pipe = transformers.pipeline(
            "text-generation",
            model=args.model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device=device,
        )
        if getattr(pipe.tokenizer, "pad_token_id", None) is None:
            eos = pipe.model.config.eos_token_id
            pipe.tokenizer.pad_token_id = eos if isinstance(eos, int) else eos[0]

        gen_config = transformers.GenerationConfig(max_new_tokens=args.max_new_tokens)
        results = inducer.induce_rules(pipe, generation_config=gen_config,
                                       batch_size=args.batch_size,
                                       disable_thinking=args.disable_thinking)

    # Save + print
    import datetime, re
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = re.sub(r"[^a-zA-Z0-9]", "-", args.model)
    default_name = f"induced_rules_{model_slug}_{timestamp}.json"
    out_path = Path(args.output if args.output else Path("exp_results") / default_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(results)} rule sets to {out_path}")

    # Quick preview
    for r in results[:3]:
        print(f"\n── Cluster {r['cluster_id']} (size={r['cluster_size']}) ──")
        print(f"   Pattern : {r['pattern_sample']}")
        print(f"   Examples: {r['addresses_sample']}")
        print(f"   Rules:\n{r['rules_text']}")


if __name__ == "__main__":
    main()
