"""
Rule induction for BZK address parsing, inspired by "LLMs can learn rules".

Induction pipeline
──────────────────
1. Compute a hybrid similarity matrix over training addresses.
   Two modes are supported (set via RuleInducer.hybrid_mode):

   "fixed"   (default)
       alpha  × cosine(emb_i, emb_j)  +  (1-alpha) × pattern_sim(i, j)
       A single global alpha blends both representations uniformly.

   "adaptive"
       w_ij × cosine(emb_i, emb_j)  +  (1-w_ij) × pattern_sim(i, j)
       where w_ij = (w_i + w_j) / 2  and  w_i = 1 − coverage_i.
       coverage_i ∈ [0,1] is the fraction of address_i characters that the
       trained spaCy NER model labels.  High NER coverage: pattern is
       reliable: w_i is small (lean on patterns).  Low coverage: the model
       did not recognise the structure → w_i is large (lean on embeddings).
       Each address thus self-selects the more informative representation,
       and pairwise similarity uses the average of the two addresses' weights.

2. Cluster with agglomerative clustering on the distance matrix (1 − sim).
3. For each non-singleton cluster, build a prompt that shows the LLM
   the annotated examples and asks it to write IF-THEN extraction rules.
4. Return the induced rules together with cluster metadata.

Quick usage: uv run python induce_rules.py --distance-threshold 0.35 --hybrid-mode adaptive
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Optional


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sentence_transformers
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import spacy
import transformers

from mllms import ner_address_to_pattern, _BZK_LABELS
from utils import compare_preds as _compare_preds

import os
from openai import OpenAI
from dotenv import load_dotenv


# 1. Similarity / distance matrices 

def _pattern_similarity_matrix(patterns: list[str]) -> np.ndarray:
    """Pairwise SequenceMatcher ratio between NER pattern strings."""
    n = len(patterns)
    sim = np.ones((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            s = SequenceMatcher(None, patterns[i], patterns[j]).ratio()
            sim[i, j] = sim[j, i] = s
    return sim


def _embedding_similarity_matrix(
    addresses: list[str],
    model: sentence_transformers.SentenceTransformer,
) -> np.ndarray:
    """Pairwise cosine similarity from sentence embeddings."""
    emb = model.encode(addresses, convert_to_tensor=True, show_progress_bar=False)
    emb = sentence_transformers.util.normalize_embeddings(emb)
    sim = (emb @ emb.T).cpu().numpy().astype(np.float32)
    return sim


def compute_hybrid_similarity(
    addresses: list[str],
    patterns: list[str],
    embedding_model: sentence_transformers.SentenceTransformer,
    alpha: float = 0.5,
) -> np.ndarray:
    """Fixed-weight hybrid: alpha × emb_sim + (1-alpha) × pattern_sim.
    """
    emb_sim = _embedding_similarity_matrix(addresses, embedding_model)
    pat_sim = _pattern_similarity_matrix(patterns)
    return np.clip(alpha * emb_sim + (1.0 - alpha) * pat_sim, 0.0, 1.0)


def _ner_coverage(address: str, nlp) -> float:
    """Fraction of non-space characters in address that are covered by a
    recognised NER entity.  Used as a confidence score for the pattern
    representation: coverage = 1: structure is well-understood by the NER
    model;  coverage = 0 : the address is opaque to the pattern extractor.
    """
    doc = nlp(address)
    labeled_chars = sum(
        len(ent.text.replace(" ", ""))
        for ent in doc.ents
        if ent.label_ in _BZK_LABELS
    )
    total_chars = len(address.replace(" ", ""))
    return labeled_chars / total_chars if total_chars > 0 else 0.0


def compute_adaptive_hybrid_similarity(
    addresses: list[str],
    patterns: list[str],
    embedding_model: sentence_transformers.SentenceTransformer,
    nlp,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive hybrid: each address self-selects its representation weight.

    Returns
    -------
    sim_matrix : np.ndarray  shape (n, n)
    emb_weights : np.ndarray  shape (n,)   per-address embedding weights
    """
    emb_sim = _embedding_similarity_matrix(addresses, embedding_model)
    pat_sim = _pattern_similarity_matrix(patterns)

    coverage = np.array(
        [_ner_coverage(addr, nlp) for addr in addresses], dtype=np.float32
    )
    # high coverage: trust patterns: low embedding weight
    emb_weights = 1.0 - coverage                         # shape (n,)

    # pairwise: average of the two addresses' weights
    w = (emb_weights[:, None] + emb_weights[None, :]) / 2.0   # (n, n)

    sim = np.clip(w * emb_sim + (1.0 - w) * pat_sim, 0.0, 1.0)
    return sim, emb_weights


def compute_hard_selection_similarity(
    addresses: list[str],
    patterns: list[str],
    embedding_model: sentence_transformers.SentenceTransformer,
    nlp,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Hard-selection hybrid: each address picks one representation.

    coverage_i > threshold : pattern representation (NER understood the structure)
    coverage_i <= threshold : embedding representation (NER is unreliable)

    Pair cases:
      both pattern  : pat_sim(i, j)
      both emb      : emb_sim(i, j)
      mixed         : (emb_sim + pat_sim) / 2   (same as soft blend at w=0.5)

    Returns
    -------
    sim_matrix : np.ndarray  shape (n, n)
    coverage   : np.ndarray  shape (n,)  per-address NER coverage
    """
    emb_sim  = _embedding_similarity_matrix(addresses, embedding_model)
    pat_sim  = _pattern_similarity_matrix(patterns)
    coverage = np.array([_ner_coverage(addr, nlp) for addr in addresses], dtype=np.float32)

    sel_pat  = coverage > threshold
    both_pat = sel_pat[:, None] & sel_pat[None, :]
    both_emb = (~sel_pat)[:, None] & (~sel_pat)[None, :]

    sim = np.where(both_pat, pat_sim,
          np.where(both_emb, emb_sim,
                   (emb_sim + pat_sim) / 2.0))
    return np.clip(sim, 0.0, 1.0), coverage


# 2. Clustering

def cluster_by_similarity(
    similarity_matrix: np.ndarray,
    n_clusters: Optional[int] = None,
    distance_threshold: float = 0.3,
) -> np.ndarray:
    """Agglomerative clustering on a precomputed similarity matrix.

    Converts similarity: distance (1 − sim) before clustering.
    Exactly one of n_clusters / distance_threshold must drive the cut.
    """
    dist = np.clip(1.0 - similarity_matrix, 0.0, 1.0)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=None if n_clusters is not None else distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    return clustering.fit_predict(dist)


# 3. Prompt construction 

_INDUCTION_SYSTEM = """\
You are an expert in historical German address parsing from World War II era \
compensation documents. Your task is to study a group of structurally similar \
addresses together with their correct entity labels, then write concise rules \
or observations that explain how to correctly parse addresses of this structural pattern.

Entity labels:
  HouseNumber – house/building number, possibly with floor (e.g. "4/II", "Block 5")
  StreetName  – street name, typically ending in -straße, -str., -weg, -gasse, -platz
  City        – city or town, possibly with qualifier (e.g. "Frankfurt/Main", "Bergen")
  District    – administrative district, often preceded by "Krs." or "Kreis"
  Region      – broader geographic region within a country, not a federal state \
(e.g. "Ostpreußen", "Schlesien", "Westfalen", "Rheinland")
  State       – federal state or administrative division (e.g. "Bayern", "Mass", "N.Y.")
  Country     – country, often in German (e.g. "Polen", "Norwegen", "U.S.A.")

What to write:
  • Rules or observations that address NON-OBVIOUS ambiguities specific to this cluster.
  • Rules can be IF-THEN statements, but may also be plain observations or notes when \
there is no clean conditional.
  • Focus on cases where a naive parser would make the wrong call: compound place names, \
ambiguous abbreviations, misleading token positions, regional vs. country labels, etc.
  • Observations about what a label is NOT are as valuable as what it IS.
  • Each rule MUST reflect a real linguistic, geographic, or documentary pattern — not \
a surface coincidence. Ask yourself: does this rule stem from a genuine historical \
convention, a grammatical structure, or an archival practice? If the answer is no, \
omit it.
  • Order rules from most to least relevant and confident: place rules that generalise \
broadly across this cluster and whose condition is unambiguous first; place uncertain \
or narrower observations last.

What NOT to write:
  • Avoid rules based on weak or generic signals (e.g., comma position or presence of “/”) without semantic context.
  • Avoid obvious or overly general patterns that apply to most addresses (e.g. "street ends in -straße implies StreetName")
  • Do NOT output any analysis, preamble, or explanation — only the numbered list.
"""

_INDUCTION_USER = """\
The following {n} addresses share a common structural pattern. \
Each is shown with its correct entity labels.

{examples}

Write rules or observations that capture how to correctly parse addresses of this \
pattern, focusing on non-obvious disambiguation decisions that a human annotator \
following the guidelines would need to know.

Output only the numbered list, nothing else.
"""


# Few-shot seed examples for the induction task itself.
# ---------------------------------------------------------------------------
_INDUCTION_FEW_SHOTS: list[tuple[list[tuple[str, dict]], str]] = [
    # Cluster: composite city names, neighborhoods, city qualifiers 
    (
        [
            # Composite city name: extract only the primary part as City
            (
                "Berlin-Marienfelde, Teichstr. 9",
                {"HouseNumber": "9", "StreetName": "Teichstr.", "City": "Berlin"},
            ),
            # Neighborhood precedes city name: city is the second token
            (
                "Radewell Halle, Saalstr. 5",
                {"HouseNumber": "5", "StreetName": "Saalstr.", "City": "Halle"},
            ),
            # Regional suffix after slash: not extracted as State/Country
            (
                "Weener/Ostfr.",
                {"City": "Weener"},
            ),
            # Qualifier that stays with the city 
            (
                "Frankfurt/Main, Voltastr. 51",
                {"HouseNumber": "51", "StreetName": "Voltastr.", "City": "Frankfurt/Main"},
            ),
        ],
        """\
1. IF a city is written as a hyphenated compound (e.g. "Berlin-Marienfelde"), \
THEN extract only the primary part before the first hyphen as City; the secondary \
part is a neighborhood.
2. IF a neighborhood or borough name appears directly before the city name without \
punctuation (e.g. "Radewell Halle"), THEN label only the final city word as City, \
not the preceding neighborhood.
3. IF a city is followed by a regional abbreviation after "/" (e.g. "Weener/Ostfr.", \
"Neuwied/Rh."), THEN label only the first part as City.
4. IF the qualifier after "/" is a well-known geographic disambiguator that \
is conventionally kept with the city (e.g. "Frankfurt/Main", "Frankfurt a.M."), \
THEN include the full token as the City value.""",
    ),
    # Cluster: State vs. District / historical region / postal code 
    (
        [
            # Krs.: District, never State
            (
                "Krs. Breslau, Hauptstr. 12",
                {"HouseNumber": "12", "StreetName": "Hauptstr.", "District": "Breslau"},
            ),
            # Historical region after slash: not State
            (
                "Prag/Böhmen, Wenzelsplatz 3",
                {"HouseNumber": "3", "StreetName": "Wenzelsplatz", "City": "Prag"},
            ),
            # Postal district code: not State
            (
                "46, Highstone Mansions, Camden Town, London N.W. 1",
                {"HouseNumber": "46", "StreetName": "Highstone Mansions", "City": "London"},
            ),
            # Geographical region (L.I.): not State
            (
                "Great Neck, L.I., N.Y., U.S.A.",
                {"City": "Great Neck", "State": "N.Y.", "Country": "U.S.A."},
            ),
            # Large city name in state position: not State
            (
                "Sydney/Australien, George Street 100",
                {"HouseNumber": "100", "StreetName": "George Street",
                 "City": "Sydney", "Country": "Australien"},
            ),
        ],
        """\
1. IF a place name is preceded by "Krs." or "Kreis", THEN label it as District; \
never assign it to State.
2. IF the component after "/" or a comma is a historical region (e.g. "Böhmen", \
"Mähren", "Galizien"), THEN do NOT label it as State.
3. IF a postal district code follows a city name (e.g. "N.W.", "N.W. 1", "S.W."), \
THEN do NOT label it as State.
4. IF a geographical sub-region abbreviation appears (e.g. "L.I." for Long Island), \
THEN do NOT label it as State; the State is the next explicit administrative unit.
5. IF a well-known large city (e.g. "Sydney", "London") appears in a position \
that might be mistaken for State, THEN label it as City, not State.""",
    ),
]


def _format_example(address: str, label_dict: dict) -> str:
    label_str = json.dumps(label_dict, ensure_ascii=False)
    return f"  Address : {address}\n  Labels  : {label_str}"


def _format_few_shot_user(examples: list[tuple[str, dict]], max_rules: int) -> str:
    examples_str = "\n\n".join(_format_example(a, ld) for a, ld in examples)
    return _INDUCTION_USER.format(
        n=len(examples), examples=examples_str, max_rules=max_rules
    )


def build_induction_messages(
    addresses: list[str],
    label_dicts: list[dict],
    max_rules: int = 5,
    few_shots: list[tuple[list[tuple[str, dict]], str]] | None = None,
) -> list[dict]:
    """Build the chat message list for one cluster's induction prompt.
    """
    if few_shots is None:
        few_shots = _INDUCTION_FEW_SHOTS

    messages: list[dict] = [{"role": "system", "content": _INDUCTION_SYSTEM}]

    for shot_examples, shot_rules in few_shots:
        messages.append({
            "role": "user",
            "content": _format_few_shot_user(shot_examples, max_rules),
        })
        messages.append({"role": "assistant", "content": shot_rules})

    messages.append({
        "role": "user",
        "content": _format_few_shot_user(list(zip(addresses, label_dicts)), max_rules),
    })
    return messages


# 4. RuleInducer class 

class RuleInducer:
    """Clusters training addresses and induces parsing rules via an LLM.
    """

    def __init__(
        self,
        addresses: pd.Series,
        labels_df: pd.DataFrame,
        labels_to_include: list[str],
        embedding_model: str | sentence_transformers.SentenceTransformer = "multi-qa-mpnet-base-dot-v1",
        ner_model_dir: str = "models/ner_bzk",
        hybrid_mode: str = "hard",
        alpha: float = 0.5,
        coverage_threshold: float = 0.5,
        distance_threshold: float = 0.3,
        n_clusters: Optional[int] = None,
        max_cluster_examples: int = 8,
        max_rules_per_cluster: int = 5,
        min_cluster_size: int = 2,
    ):
        if hybrid_mode not in ("fixed", "adaptive", "hard"):
            raise ValueError(f"hybrid_mode must be 'fixed', 'adaptive', or 'hard', got '{hybrid_mode}'")

        self.addresses = addresses.reset_index(drop=True)
        self.labels_df = labels_df[labels_to_include].reset_index(drop=True)
        self.labels_to_include = labels_to_include
        self.hybrid_mode = hybrid_mode
        self.alpha = alpha
        self.coverage_threshold = coverage_threshold
        self.max_cluster_examples = max_cluster_examples
        self.max_rules_per_cluster = max_rules_per_cluster
        self.min_cluster_size = min_cluster_size

        if isinstance(embedding_model, str):
            print(f"Loading embedding model '{embedding_model}'…")
            self.embedding_model = sentence_transformers.SentenceTransformer(embedding_model)
        else:
            self.embedding_model = embedding_model

        print(f"Loading NER model from '{ner_model_dir}'…")
        self._nlp = spacy.load(ner_model_dir)

        addr_list = list(self.addresses)

        print(f"Computing NER patterns for {len(addr_list)} addresses…")
        self.patterns: list[str] = [
            ner_address_to_pattern(a, self._nlp) for a in addr_list
        ]

        print(f"Computing hybrid similarity matrix (mode='{hybrid_mode}')…")
        if hybrid_mode == "adaptive":
            self.sim_matrix, self.emb_weights = compute_adaptive_hybrid_similarity(
                addr_list, self.patterns, self.embedding_model, self._nlp
            )
        elif hybrid_mode == "hard":
            self.sim_matrix, coverage = compute_hard_selection_similarity(
                addr_list, self.patterns, self.embedding_model, self._nlp,
                threshold=coverage_threshold,
            )
            self.emb_weights = 1.0 - coverage
        else:
            self.sim_matrix = compute_hybrid_similarity(
                addr_list, self.patterns, self.embedding_model, alpha=alpha
            )
            self.emb_weights = np.full(len(addr_list), alpha, dtype=np.float32)

        print("Clustering…")
        self.cluster_ids = cluster_by_similarity(
            self.sim_matrix,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
        )
        unique = sorted(set(self.cluster_ids))
        sizes = {cid: int((self.cluster_ids == cid).sum()) for cid in unique}
        print(
            f"Found {len(unique)} clusters  "
            f"(sizes: min={min(sizes.values())}, "
            f"max={max(sizes.values())}, "
            f"mean={np.mean(list(sizes.values())):.1f})"
        )

    # helpers 

    def _cluster_indices(self, cid: int) -> np.ndarray:
        """Return indices of members of cluster cid, sorted by centrality."""
        idx = np.where(self.cluster_ids == cid)[0]
        if len(idx) > 1:
            sub = self.sim_matrix[np.ix_(idx, idx)]
            centrality = sub.mean(axis=1)
            idx = idx[np.argsort(centrality)[::-1]]
        return idx

    def _cluster_data(self, cid: int) -> tuple[list[str], list[dict]]:
        """Return (addresses, label_dicts) for a cluster.

        Indices are ordered by intra-cluster centrality.  Duplicate address
        strings are removed (first/most-central occurrence wins) before
        capping at max_cluster_examples, so the induction prompt receives
        diverse examples rather than the same address repeated many times.
        """
        idx = self._cluster_indices(cid)

        # Deduplicate by normalised address string, preserving centrality order
        seen: set[str] = set()
        unique_idx = []
        for i in idx:
            key = self.addresses[i].strip().lower()
            if key not in seen:
                seen.add(key)
                unique_idx.append(i)

        unique_idx = unique_idx[: self.max_cluster_examples]
        addrs = [self.addresses[i] for i in unique_idx]
        label_dicts = [
            {k: v for k, v in self.labels_df.iloc[i].items() if pd.notna(v) and v != ""}
            for i in unique_idx
        ]
        return addrs, label_dicts


    def cluster_summary(self) -> pd.DataFrame:
        """Return a DataFrame with one row per cluster.

        Columns include id, size, pattern sample, and (in adaptive mode) the
        mean embedding weight of cluster members — useful for understanding
        whether a cluster was shaped more by embeddings or patterns.
        """
        rows = []
        for cid in sorted(set(self.cluster_ids)):
            idx = self._cluster_indices(cid)
            row = {
                "cluster_id":      int(cid),
                "size":            len(idx),
                "pattern_sample":  self.patterns[idx[0]],
                "address_sample":  self.addresses[idx[0]],
                "mean_emb_weight": float(self.emb_weights[idx].mean()),
            }
            rows.append(row)
        return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)

    def save_clusters(self, path: str = "clusters.csv") -> pd.DataFrame:
        """Save per-address cluster assignments to a CSV file.

        Columns: index, address, pattern, cluster_id, cluster_size, emb_weight.
        Returns the DataFrame so it can be inspected inline.
        """
        sizes = {
            cid: int((self.cluster_ids == cid).sum())
            for cid in set(self.cluster_ids)
        }
        df = pd.DataFrame({
            "address":      self.addresses,
            "pattern":      self.patterns,
            "cluster_id":   self.cluster_ids,
            "cluster_size": [sizes[c] for c in self.cluster_ids],
            "emb_weight":   self.emb_weights,
        })
        df.to_csv(path, index=True, index_label="idx")
        print(f"Cluster assignments saved to {path}  ({len(df)} rows)")
        return df

    def plot_clusters(
        self,
        path: str | None = "clusters.png",
        max_labels: int = 30,
        figsize: tuple[int, int] = (14, 10),
        tsne_kwargs: dict | None = None,
    ) -> None:
        """2-D t-SNE scatter plot of the cluster assignments.

        Points are coloured by cluster id.  The most central address of each
        cluster (up to max_labels largest clusters) is annotated with its
        NER pattern.
        """

        dist = np.clip(1.0 - self.sim_matrix, 0.0, 1.0)
        kw = dict(n_components=2, metric="precomputed",
                  init="random", random_state=42, perplexity=min(30, len(dist) - 1))
        if tsne_kwargs:
            kw.update(tsne_kwargs)
        print("Running t-SNE…")
        coords = TSNE(**kw).fit_transform(dist)   # (n, 2)

        # Map cluster ids to a compact colour index
        unique_ids = sorted(set(self.cluster_ids))
        cmap = plt.cm.get_cmap("tab20", min(len(unique_ids), 20))
        id_to_color = {cid: cmap(i % 20) for i, cid in enumerate(unique_ids)}
        colors = [id_to_color[c] for c in self.cluster_ids]

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=18, alpha=0.7, linewidths=0)

        # Annotate centroid of the largest clusters
        sizes = {cid: int((self.cluster_ids == cid).sum()) for cid in unique_ids}
        top_clusters = sorted(unique_ids, key=lambda c: sizes[c], reverse=True)[:max_labels]
        for cid in top_clusters:
            idx = self._cluster_indices(cid)   # sorted by centrality
            cx, cy = coords[idx[0], 0], coords[idx[0], 1]
            label = self.patterns[idx[0]]
            if len(label) > 35:
                label = label[:33] + "…"
            ax.annotate(
                f"[{sizes[cid]}] {label}",
                xy=(cx, cy), fontsize=6.5,
                xytext=(4, 4), textcoords="offset points",
            )

        ax.set_title(
            f"Address clusters (n={len(self.addresses)}, "
            f"k={len(unique_ids)}, mode={self.hybrid_mode})",
            fontsize=11,
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.axis("off")
        fig.tight_layout()

        if path:
            fig.savefig(path, dpi=150)
            print(f"Cluster plot saved to {path}")
        else:
            plt.show()
        plt.close(fig)

    def build_all_prompts(
        self,
        few_shots: list[tuple[list[tuple[str, dict]], str]] | None = None,
    ) -> list[tuple[int, list[dict]]]:
        """Return (cluster_id, chat_messages) pairs for all qualifying clusters.
        """
        prompts = []
        for cid in sorted(set(self.cluster_ids)):
            idx = self._cluster_indices(cid)
            if len(idx) < self.min_cluster_size:
                continue
            addrs, label_dicts = self._cluster_data(cid)
            msgs = build_induction_messages(
                addrs, label_dicts,
                max_rules=self.max_rules_per_cluster,
                few_shots=few_shots,
            )
            prompts.append((cid, msgs))
        return prompts

    def induce_rules(
        self,
        llm_pipeline,
        generation_config=None,
        batch_size: int = 4,
        few_shots: list[tuple[list[tuple[str, dict]], str]] | None = None,
        disable_thinking: bool = False,
    ) -> list[dict]:
        """Run the induction step for all qualifying clusters.
        """

        if generation_config is None:
            generation_config = transformers.GenerationConfig(max_new_tokens=512)

        prompts_data = self.build_all_prompts(few_shots=few_shots)
        print(f"Inducing rules for {len(prompts_data)} clusters (skipped singletons)…")

        results = []
        for start in range(0, len(prompts_data), batch_size):
            batch = prompts_data[start : start + batch_size]
            conversations = [msgs for _, msgs in batch]

            if disable_thinking:
                conversations = llm_pipeline.tokenizer.apply_chat_template(
                    conversations,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

            outputs = llm_pipeline(
                conversations,
                generation_config=generation_config,
            )
            for (cid, _), output in zip(batch, outputs):
                generated = output[0]["generated_text"]
                if disable_thinking:
                    response = generated.split("<|im_start|>assistant")[-1]
                elif isinstance(generated, list):
                    # Last message is the assistant turn
                    response = generated[-1]["content"]
                else:
                    response = str(generated)

                addrs, label_dicts = self._cluster_data(cid)
                results.append({
                    "cluster_id":      int(cid),
                    "cluster_size":    int((self.cluster_ids == cid).sum()),
                    "pattern_sample":  self.patterns[self._cluster_indices(cid)[0]],
                    "addresses_sample": addrs[:3],
                    "labels_sample":   label_dicts[:3],
                    "rules_text":      response.strip(),
                })
            print(
                f"  [{start + len(batch)}/{len(prompts_data)}] clusters processed"
            )

        return results

    def induce_rules_openai(
        self,
        model: str = "gpt-4.1-mini",
        few_shots: list[tuple[list[tuple[str, dict]], str]] | None = None,
        max_tokens: int = 512,
    ) -> list[dict]:
        """Run rule induction using an OpenAI chat model.

        Reads OPENAI_API from the .env file.  Sends one
        request per qualifying cluster.

        """
      

        load_dotenv()
        api_key = os.getenv("OPENAI_API") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API key not found in .env or environment")

        client = OpenAI(api_key=api_key)
        prompts_data = self.build_all_prompts(few_shots=few_shots)
        print(f"Inducing rules for {len(prompts_data)} clusters via {model}…")

        results = []
        total_input = total_output = 0

        for i, (cid, messages) in enumerate(prompts_data, 1):
            # gpt-5-mini and o-series models use max_completion_tokens;
            # older models use max_tokens.
            _NEW_TOKEN_PARAM_MODELS = ("gpt-5", "o1", "o3", "o4")
            token_param = (
                "max_completion_tokens"
                if any(model.startswith(p) for p in _NEW_TOKEN_PARAM_MODELS)
                else "max_tokens"
            )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **{token_param: max_tokens},
            )
            rules_text = response.choices[0].message.content
            total_input  += response.usage.prompt_tokens
            total_output += response.usage.completion_tokens

            addrs, label_dicts = self._cluster_data(cid)
            results.append({
                "cluster_id":       int(cid),
                "cluster_size":     int((self.cluster_ids == cid).sum()),
                "pattern_sample":   self.patterns[self._cluster_indices(cid)[0]],
                "addresses_sample": addrs[:3],
                "labels_sample":    label_dicts[:3],
                "rules_text":       rules_text,
            })
            print(f"  [{i}/{len(prompts_data)}] cluster {cid} done")

        print(f"\nTotal tokens — input: {total_input:,}  output: {total_output:,}")
        return results


# . RuleDeductor

class RuleDeductor:
    """Maps a new unseen address to its closest cluster and returns the
    induced rules for that cluster.

    Cluster assignment reuses the same hybrid similarity used during induction

    Only clusters that have an entry in induced_rules are considered.
    """

    def __init__(
        self,
        inducer: "RuleInducer",
        induced_rules: list[dict],
        min_similarity: float = 0.0,
    ):
        self.inducer = inducer
        self.min_similarity = min_similarity

        # Index rules and stored examples by cluster id
        self._rules: dict[int, str] = {
            r["cluster_id"]: r["rules_text"]
            for r in induced_rules
            if r.get("rules_text", "").strip()
        }
        # addresses_sample / labels_sample were saved during induction; reuse them
        # so inference never has to recompute cluster similarity from scratch.
        self._examples: dict[int, list[tuple[str, dict]]] = {
            r["cluster_id"]: list(zip(r.get("addresses_sample", []),
                                       r.get("labels_sample", [])))
            for r in induced_rules
        }

        # Pre-compute representative embedding + pattern for each rule cluster.
        #
        # The induced_rules JSON is self-contained: it stores addresses_sample
        # and pattern_sample for every cluster.  We use those directly as the
        # cluster representative — no lookup into the current inducer's cluster
        # assignments is needed, making the deductor robust to re-clustering.
        self._rep_cids: list[int] = []
        self._rep_patterns: list[str] = []
        rep_addresses: list[str] = []

        for rule_record in induced_rules:
            rule_key = rule_record["cluster_id"]
            if rule_key not in self._rules:
                continue  # no non-empty rules_text for this record

            sample_addrs = rule_record.get("addresses_sample", [])
            if not sample_addrs:
                print(f"  [RuleDeductor] rule cluster {rule_key} has no addresses_sample — skipping")
                continue

            pattern = rule_record.get("pattern_sample", "")
            if not pattern:
                pattern = ner_address_to_pattern(sample_addrs[0], inducer._nlp)

            self._rep_cids.append(rule_key)
            self._rep_patterns.append(pattern)
            rep_addresses.append(sample_addrs[0])

        if not rep_addresses:
            raise ValueError(
                "RuleDeductor: no rule clusters with non-empty rules_text and addresses_sample found."
            )

        emb = inducer.embedding_model.encode(
            rep_addresses, convert_to_tensor=True, show_progress_bar=False
        )
        self._rep_embeddings = sentence_transformers.util.normalize_embeddings(emb)  # (k, d)

    # internal 

    def _scores(self, address: str) -> np.ndarray:
        """Return hybrid similarity scores between address and every cluster rep."""

        # Embedding similarity
        emb = self.inducer.embedding_model.encode(
            [address], convert_to_tensor=True, show_progress_bar=False
        )
        emb = sentence_transformers.util.normalize_embeddings(emb)               # (1, d)
        emb_scores = (emb @ self._rep_embeddings.T).squeeze(0).cpu().numpy()

        # Pattern similarity
        pat = ner_address_to_pattern(address, self.inducer._nlp)
        pat_scores = np.array([
            SequenceMatcher(None, pat, rp).ratio()
            for rp in self._rep_patterns
        ], dtype=np.float32)

        # Blend weight 
        if self.inducer.hybrid_mode in ("adaptive", "hard"):
            w = 1.0 - _ner_coverage(address, self.inducer._nlp)
        else:
            w = self.inducer.alpha

        return w * emb_scores + (1.0 - w) * pat_scores

   
    def assign(self, address: str) -> tuple[int | None, float]:
        """Return (cluster_id, similarity) for the best matching cluster.

        Returns (None, score) when the best score is below min_similarity.
        """
        scores = self._scores(address)
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score < self.min_similarity:
            return None, score
        return self._rep_cids[best], score

    def get_rules(self, address: str) -> str:
        """Return the induced rules text for the best matching cluster.

        Returns an empty string when no cluster exceeds min_similarity or the
        matched cluster has no rules.
        """
        cid, _ = self.assign(address)
        if cid is None:
            return ""
        return self._rules.get(cid, "")

    def get_examples(self, address: str) -> list[tuple[str, dict]]:
        """Return the stored cluster examples for the best matching cluster.

        These are the (address, label_dict) pairs saved during induction, so
        no similarity recomputation against the full training set is needed.
        Returns an empty list when no cluster exceeds min_similarity.
        """
        cid, _ = self.assign(address)
        if cid is None:
            return []
        return self._examples.get(cid, [])


# 5. RuleEvaluator

class RuleEvaluator:
    """Evaluates the impact of induced rules on parsing accuracy.

    Supports four complementary analyses:

    1. A/B Test            — overall metric comparison (with vs without rules)
    2. per_label A/B Test  — column-wise precision / recall / F1 / accuracy
    3. per_cluster_metrics — per-cluster A/B with accuracy and F1 lift
    4. rule_importance     — clusters ranked by size-weighted accuracy lift
    """

    def __init__(
        self,
        inducer: "RuleInducer",
        induced_rules: list[dict],
        addresses: pd.Series,
        ground_truth: pd.DataFrame,
        labels_to_include: list[str],
        min_similarity: float = 0.0,
    ):
        self._compare_preds = _compare_preds

        self.inducer = inducer
        self.induced_rules = induced_rules
        self.addresses = addresses.reset_index(drop=True)
        self.ground_truth = ground_truth[labels_to_include].reset_index(drop=True)
        self.labels = labels_to_include

        self.deductor = RuleDeductor(inducer, induced_rules, min_similarity=min_similarity)

        print(f"Assigning {len(self.addresses)} eval addresses to rule clusters…")
        assignments = [self.deductor.assign(addr) for addr in self.addresses]
        # None for unmatched addresses
        self.cluster_ids = pd.array(
            [cid for cid, _ in assignments], dtype=pd.Int64Dtype()
        )
        self.cluster_scores = np.array([score for _, score in assignments], dtype=np.float32)

        self._rules_map: dict[int, str] = {
            r["cluster_id"]: r["rules_text"] for r in induced_rules
        }
        self._pattern_map: dict[int, str] = {
            r["cluster_id"]: r.get("pattern_sample", "") for r in induced_rules
        }

    def _to_df(self, preds) -> pd.DataFrame:
        if isinstance(preds, list):
            preds = pd.DataFrame(preds)
        return preds.reset_index(drop=True)

    def _col_metrics(
        self,
        preds_df: pd.DataFrame,
        mask: "np.ndarray | None" = None,
    ) -> pd.DataFrame:
        """Per-label exact-match accuracy, precision, recall, F1.

        ``mask`` is a boolean array selecting a subset of eval rows 
        """
        gt = self.ground_truth if mask is None else self.ground_truth.loc[mask]
        pr = preds_df if mask is None else preds_df.loc[mask]
        gt = gt.reset_index(drop=True)
        pr = pr.reset_index(drop=True)

        rows = []
        for col in self.labels:
            gt_col = gt[col].fillna("") if col in gt.columns else pd.Series([""] * len(gt))
            pr_col = pr[col].fillna("") if col in pr.columns else pd.Series([""] * len(pr))

            has_label = gt_col != ""
            has_pred  = pr_col != ""
            exact     = gt_col == pr_col

            tp = int((exact & has_label & has_pred).sum())
            fp = int((has_pred & ~exact).sum())
            fn = int((has_label & ~exact).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            f1 = (
                2 * precision * recall / (precision + recall)
                if (not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0)
                else float("nan")
            )
            accuracy = float(exact.sum()) / len(gt_col) if len(gt_col) > 0 else float("nan")

            rows.append({
                "label":     col,
                "accuracy":  accuracy,
                "precision": precision,
                "recall":    recall,
                "f1":        f1,
                "support":   int(has_label.sum()),
            })
        return pd.DataFrame(rows).set_index("label")

   

    def ab_test(
        self,
        preds_with: "list[dict] | pd.DataFrame",
        preds_without: "list[dict] | pd.DataFrame",
        columns: "list[str] | None" = None,
    ) -> pd.DataFrame:
        """Overall A/B comparison between predictions made with and without rules.

        Uses the project-standard metric suite (accuracy,
        precision, recall, F1, Levenshtein similarity, tolerance bands).
        """
        cols = columns if columns is not None else self.labels
        with_df    = self._to_df(preds_with)
        without_df = self._to_df(preds_without)

        m_with    = self._compare_preds(with_df,    self.ground_truth, cols)
        m_without = self._compare_preds(without_df, self.ground_truth, cols)

        rows = []
        for key in m_with:
            rows.append({
                "metric":        key,
                "with_rules":    m_with[key],
                "without_rules": m_without[key],
                "delta":         m_with[key] - m_without[key],
            })
        return pd.DataFrame(rows).set_index("metric")

    def per_label_ab_test(
        self,
        preds_with: "list[dict] | pd.DataFrame",
        preds_without: "list[dict] | pd.DataFrame",
    ) -> pd.DataFrame:
        """Per-label A/B test: accuracy, precision, recall, F1 for both conditions.
        """
        with_df    = self._to_df(preds_with)
        without_df = self._to_df(preds_without)

        with_m    = self._col_metrics(with_df)
        without_m = self._col_metrics(without_df)

        result = pd.DataFrame(index=with_m.index)
        for metric in ("accuracy", "precision", "recall", "f1"):
            result[f"{metric}_with"]    = with_m[metric]
            result[f"{metric}_without"] = without_m[metric]
            result[f"{metric}_delta"]   = with_m[metric] - without_m[metric]
        result["support"] = with_m["support"]
        return result

    def per_cluster_metrics(
        self,
        preds_with: "list[dict] | pd.DataFrame",
        preds_without: "list[dict] | pd.DataFrame",
        min_cluster_eval_size: int = 3,
    ) -> pd.DataFrame:
        """Per-cluster A/B: accuracy and macro-F1 lift for every rule cluster.

        Only clusters where at least min_cluster_eval_size addresses
        were matched are included.
        """
        with_df    = self._to_df(preds_with)
        without_df = self._to_df(preds_without)

        train_sizes = {
            int(cid): int((self.inducer.cluster_ids == cid).sum())
            for cid in set(self.inducer.cluster_ids)
        }

        rows = []
        for cid in sorted(self._rules_map):
            mask = np.array(self.cluster_ids == cid, dtype=bool)
            n_eval = int(mask.sum())
            if n_eval < min_cluster_eval_size:
                continue

            with_m    = self._col_metrics(with_df,    mask)
            without_m = self._col_metrics(without_df, mask)

            # Macro-average over labels that have at least one support example
            supported = with_m["support"] > 0
            with_f1    = float(with_m.loc[supported, "f1"].mean())
            without_f1 = float(without_m.loc[supported, "f1"].mean())
            with_acc    = float(with_m["accuracy"].mean())
            without_acc = float(without_m["accuracy"].mean())

            rows.append({
                "cluster_id":       cid,
                "size_train":       train_sizes.get(cid, 0),
                "size_eval":        n_eval,
                "pattern_sample":   self._pattern_map.get(cid, ""),
                "accuracy_with":    round(with_acc,    4),
                "accuracy_without": round(without_acc, 4),
                "accuracy_lift":    round(with_acc - without_acc, 4),
                "f1_with":          round(with_f1,    4),
                "f1_without":       round(without_f1, 4),
                "f1_lift":          round(with_f1 - without_f1, 4),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("accuracy_lift", ascending=False)
            .reset_index(drop=True)
        )

    def cluster_importance(
        self,
        preds_with: "list[dict] | pd.DataFrame",
        preds_without: "list[dict] | pd.DataFrame",
        min_cluster_eval_size: int = 3,
    ) -> pd.DataFrame:
        """Cluster-level importance: clusters ranked by size-weighted accuracy lift.

        Compares preds_with (all rules injected) vs preds_without (no rules)
        at the cluster level.  This tells which clusters benefit or hurt from having their rules injected
        as a block.  

        weighted_contribution = accuracy_lift × (size_eval / total_eval).
        Positive values mean the cluster's rules help overall accuracy; negative
        means they hurt.  The rank index is 1-based.
        """
        cluster_df = self.per_cluster_metrics(
            preds_with, preds_without, min_cluster_eval_size
        )

        if cluster_df.empty:
            return cluster_df

        total_eval = cluster_df["size_eval"].sum()
        cluster_df = cluster_df.copy()
        cluster_df["weighted_contribution"] = (
            cluster_df["accuracy_lift"] * cluster_df["size_eval"] / total_eval
        ).round(4)
        cluster_df["rules_text"] = cluster_df["cluster_id"].map(self._rules_map)

        result = (
            cluster_df[
                ["cluster_id", "size_eval", "accuracy_lift",
                 "weighted_contribution", "pattern_sample", "rules_text"]
            ]
            .sort_values("weighted_contribution", ascending=False)
            .reset_index(drop=True)
        )
        result.index = result.index + 1
        result.index.name = "rank"
        return result

    # --- individual-rule evaluation ---

    @staticmethod
    def _split_rules(rules_text: str) -> list[str]:
        """Split a numbered rules block into individual rule strings.

        Handles the two formats the LLM produces:
          "1. IF …\\n2. IF …"      (newline-separated)
          "1. IF … 2. IF …"        (space-separated, no newlines)
        Each returned string retains its full text but without the leading
        number+period prefix.
        """
        import re
        # Split on a digit(s) + period that either starts the string or follows
        # a newline / whitespace boundary.
        parts = re.split(r'(?:^|\n)\s*\d+\.\s+', rules_text.strip())
        # If that produced only one chunk the rules may be on a single line
        if len(parts) <= 1:
            parts = re.split(r'\s+\d+\.\s+', rules_text.strip())
        return [p.strip() for p in parts if p.strip()]

    def evaluate_rules_individually(
        self,
        parse_fn: "callable[[list[str], str], list[dict]]",
        preds_with: "list[dict] | pd.DataFrame",
        min_cluster_eval_size: int = 3,
    ) -> pd.DataFrame:
        """Evaluate each rule via leave-one-out ablation.

        For every rule in every cluster, the model is run with all rules
        except that one.  The result is compared against preds_with
        (all rules present).  A drop in accuracy when a rule is removed
        means it was contributing positively; an improvement means it was
        hurting.

        accuracy_lift is defined as:
            accuracy(all rules) − accuracy(all rules except this one)
        so positive = rule helps, negative = rule hurts.
        """
        with_df = self._to_df(preds_with)
        rows = []

        for cid, full_rules_text in self._rules_map.items():
            mask = np.array(self.cluster_ids == cid, dtype=bool)
            n_eval = int(mask.sum())
            if n_eval < min_cluster_eval_size:
                continue

            cluster_addrs = list(self.addresses[mask])
            individual_rules = self._split_rules(full_rules_text)
            if len(individual_rules) < 2:
                # LOO requires at least 2 rules; single-rule clusters are skipped
                # because removing the only rule is equivalent to the no-rules baseline
                # which is already captured by cluster_importance().
                continue

            # Full-rules baseline for this cluster
            base_m = self._col_metrics(with_df, mask)
            supported = base_m["support"] > 0
            base_acc = float(base_m["accuracy"].mean())
            base_f1  = float(base_m.loc[supported, "f1"].mean()) if supported.any() else float("nan")

            # Row-level correct/wrong counts under the full-rules condition
            # (i.e. when this rule IS active — used as a signal for both metrics).
            gt_sub = self.ground_truth.loc[mask].reset_index(drop=True)
            with_sub = with_df.loc[mask].reset_index(drop=True)
            row_all_correct = pd.Series([True] * n_eval)
            for col in self.labels:
                gt_c = gt_sub[col].fillna("") if col in gt_sub.columns else pd.Series([""] * n_eval)
                pr_c = with_sub[col].fillna("") if col in with_sub.columns else pd.Series([""] * n_eval)
                row_all_correct &= (gt_c == pr_c)
            n_correct_with = int(row_all_correct.sum())
            n_wrong_with   = n_eval - n_correct_with

            for rule_idx, rule_text in enumerate(individual_rules, start=1):
                # All rules except the current one
                others = [r for i, r in enumerate(individual_rules, start=1) if i != rule_idx]
                injected_text = "\n".join(f"{i}. {r}" for i, r in enumerate(others, start=1))

                try:
                    cluster_preds = parse_fn(cluster_addrs, injected_text)
                except Exception as exc:
                    print(f"  parse_fn failed for cluster {cid} rule {rule_idx}: {exc}")
                    continue

                pred_df = self._to_df(cluster_preds)
                # _col_metrics compares against self.ground_truth with a global mask,
                # but parse_fn returns only the cluster's addresses, so we compare
                # against the cluster ground-truth slice directly (gt_sub already computed above).
                local_rows = []
                for col in self.labels:
                    gt_col = gt_sub[col].fillna("") if col in gt_sub.columns else pd.Series([""] * len(gt_sub))
                    pr_col = pred_df[col].fillna("") if col in pred_df.columns else pd.Series([""] * len(pred_df))
                    has_label = gt_col != ""
                    has_pred  = pr_col != ""
                    exact     = gt_col == pr_col
                    tp = int((exact & has_label & has_pred).sum())
                    fp = int((has_pred & ~exact).sum())
                    fn = int((has_label & ~exact).sum())
                    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
                    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                    f1_v = (2 * prec * rec / (prec + rec)
                            if not (np.isnan(prec) or np.isnan(rec)) and (prec + rec) > 0
                            else float("nan"))
                    acc_v = float(exact.sum()) / len(gt_col) if len(gt_col) > 0 else float("nan")
                    local_rows.append({"label": col, "accuracy": acc_v, "f1": f1_v,
                                       "support": int(has_label.sum())})
                local_m = pd.DataFrame(local_rows).set_index("label")

                supported_local = local_m["support"] > 0
                rule_acc = float(local_m["accuracy"].mean())
                rule_f1  = float(local_m.loc[supported_local, "f1"].mean()) if supported_local.any() else float("nan")

                rows.append({
                    "cluster_id":        cid,
                    "rule_rank":         rule_idx,
                    "rule_text":         rule_text,
                    "n_eval":            n_eval,
                    "n_correct_with":    n_correct_with,
                    "n_wrong_with":      n_wrong_with,
                    "accuracy":          round(rule_acc, 4),
                    "accuracy_lift":     round(rule_acc - base_acc, 4),
                    "f1_macro":          round(rule_f1, 4),
                    "f1_lift":           round(rule_f1 - base_f1, 4),
                    "per_label_accuracy": {
                        col: round(local_m.loc[col, "accuracy"], 4)
                        for col in self.labels
                        if col in local_m.index
                    },
                })
                print(
                    f"  cluster {cid:3d} | rule {rule_idx}/{len(individual_rules)} | "
                    f"acc={rule_acc:.4f} (lift={rule_acc - base_acc:+.4f})"
                )

        return (
            pd.DataFrame(rows)
            .sort_values("accuracy_lift", ascending=False)
            .reset_index(drop=True)
        )

    def filter_rules(
        self,
        parse_fn: "callable[[list[str], str], list[dict]]",
        preds_with: "list[dict] | pd.DataFrame",
        min_accuracy_lift: float = 0.0,
        min_f1_lift: float | None = None,
        min_cluster_eval_size: int = 3,
    ) -> tuple[list[dict], pd.DataFrame]:
        """Remove individual rules that do not meet the LOO lift thresholds.

        Runs leave-one-out evaluation and drops any rule whose removal does
        not decrease accuracy by at least ``min_accuracy_lift`` (i.e. rules
        that don't contribute are pruned).
        """
        eval_df = self.evaluate_rules_individually(
            parse_fn, preds_with,
            min_cluster_eval_size=min_cluster_eval_size,
        )

        # Determine which rules pass the thresholds
        keep_mask = eval_df["accuracy_lift"] >= min_accuracy_lift
        if min_f1_lift is not None:
            keep_mask &= eval_df["f1_lift"] >= min_f1_lift
        eval_df = eval_df.copy()
        eval_df["kept"] = keep_mask

        # Build a set of (cluster_id, rule_rank) to keep
        kept_pairs: set[tuple[int, int]] = set(
            zip(
                eval_df.loc[keep_mask, "cluster_id"],
                eval_df.loc[keep_mask, "rule_rank"],
            )
        )

        # Rebuild induced_rules with filtered rules_text
        filtered_rules = []
        for rule_record in self.induced_rules:
            cid = rule_record["cluster_id"]
            record = dict(rule_record)  # shallow copy

            individual = self._split_rules(rule_record.get("rules_text", ""))
            kept_rules = [
                rule_text
                for rank, rule_text in enumerate(individual, start=1)
                if (cid, rank) in kept_pairs
            ]

            if kept_rules:
                record["rules_text"] = "\n".join(
                    f"{i}. {r}" for i, r in enumerate(kept_rules, start=1)
                )
            else:
                record["rules_text"] = ""

            n_before = len(individual)
            n_after  = len(kept_rules)
            if n_after < n_before:
                print(
                    f"  cluster {cid}: kept {n_after}/{n_before} rules "
                    f"(threshold acc_lift >= {min_accuracy_lift})"
                )
            filtered_rules.append(record)

        return filtered_rules, eval_df

    def summary(
        self,
        preds_with: "list[dict] | pd.DataFrame",
        preds_without: "list[dict] | pd.DataFrame",
        top_n: int = 5,
    ) -> None:
        """Print a concise evaluation summary to stdout."""
        ab  = self.ab_test(preds_with, preds_without)
        imp = self.cluster_importance(preds_with, preds_without)

        covered = int(pd.notna(pd.array(self.cluster_ids, dtype=pd.Int64Dtype())).sum())
        print(f"\n{'='*60}")
        print(f"Rule Evaluation Summary")
        print(f"{'='*60}")
        print(f"Eval addresses  : {len(self.addresses)}")
        print(f"Covered by rules: {covered} ({100 * covered / len(self.addresses):.1f}%)")

        print(f"\n--- Overall A/B ---")
        for metric in ("accuracy", "precision", "recall", "f1"):
            if metric in ab.index:
                row = ab.loc[metric]
                sign = "+" if row["delta"] >= 0 else ""
                print(
                    f"  {metric:12s}: {row['with_rules']:.4f} vs "
                    f"{row['without_rules']:.4f}  ({sign}{row['delta']:.4f})"
                )

        print(f"\n--- Top {top_n} rule clusters by weighted contribution ---")
        if not imp.empty:
            for rank, row in imp.head(top_n).iterrows():
                sign = "+" if row["weighted_contribution"] >= 0 else ""
                print(
                    f"  #{rank} cluster {int(row['cluster_id']):3d} | "
                    f"eval={int(row['size_eval']):3d} | "
                    f"lift={row['accuracy_lift']:+.4f} | "
                    f"contribution={sign}{row['weighted_contribution']:.4f}"
                )
                print(f"     pattern: {str(row['pattern_sample'])[:60]}")
        print(f"{'='*60}\n")
