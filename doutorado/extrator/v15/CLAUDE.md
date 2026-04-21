# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OpenIE-PT v15** is a Portuguese Open Information Extraction (OpenIE) research system. It extracts relation triplets `(arg1, rel, arg2)` from Portuguese sentences aligned with the **BIA (Banco de Interpretações Agregadas)** annotation standard. The system combines:

- **Universal Dependencies (UD) parsing** via Stanza for structural candidate generation
- **BERTimbau** (neuralmind/bert-base-portuguese-cased) attention heads for scoring and reranking candidates
- **Theoretical validation rules** (E1–E9 extraction rules, S1–S5 span rules) for filtering/annotating extractions
- **Ablation studies** across 29 experiment configurations answering research questions (RQ1–RQ6)

## Running the Extractor

```bash
python openie_v11_bia_theoretical_experiment_selfcontained_patched.py \
  --gold bia_gold_sentences.jsonl \
  --output-dir ablation_experiments/my_experiment \
  --dataset-name my_experiment \
  [--bert-model neuralmind/bert-base-portuguese-cased] \
  [--theory-mode off|annotate|filter] \
  [--heads-mode rank|all|forced] \
  [--attn-threshold 0.0] \
  [--no-attn] \
  [--s2-span-policy minimal|extensive|both] \
  [--strict-theory] \
  [--block-reported-belief] \
  [--no-e8-clausal]
```

Key flags:
- `--theory-mode filter` — remove candidates violating E/S rules; `annotate` tags them without removing; `off` disables rule checking
- `--no-attn` — skip BERT attention scoring entirely (UD-only baseline)
- `--heads-mode rank` — use top-ranked heads from Bosque corpus; `all` uses all heads; `forced` uses a fixed set

## Architecture

Everything is in a single self-contained file: [openie_v11_bia_theoretical_experiment_selfcontained_patched.py](openie_v11_bia_theoretical_experiment_selfcontained_patched.py) (~3388 lines). The Jupyter notebook [openie_pt_experiment_final.ipynb](openie_pt_experiment_final.ipynb) is used for visualization and analysis of results.

### Pipeline Stages (in order)

1. **Initialization** — Load Stanza parser, BERTimbau model, and rank attention heads from `pt_bosque-ud-train.conllu`
2. **UD Parsing** — Stanza produces Universal Dependencies structure per sentence
3. **Candidate Generation** — Rule-based extraction using UD dependency relations (nsubj, obj, iobj, obl, copula, coordination, relative clauses, passives, modals)
4. **Theory Validation** — Apply E-rules and S-rules per candidate; compute verdicts (True/False/None per rule)
5. **Attention Scoring** — Map UD tokens to BERT word pieces, extract attention weights, score each arg pair
6. **Theory Policy** — Apply `theory-mode`: filter out invalid, annotate, or skip
7. **Reranking** — Optionally keep top candidate per `(verb, pattern_ud)` group by attention score
8. **Deduplication + Output** — Write enriched JSONL/CSV files, metrics JSON, and summary files

### Key Classes

| Class | Location | Purpose |
|---|---|---|
| `Config` | line ~50, extended ~1880 | All extraction and experiment parameters |
| `OpenIEExtractorBIA` | line ~419, refactored ~1999 | Main extraction engine |
| `CandidateTriple` | line ~1930 | Extraction candidate with all metadata |
| `TheoryValidation` | line ~1968 | Rule verdict container per candidate |
| `UDTok` | line ~147 | UD token representation |

### Data Flow

```
bia_gold_sentences.jsonl  →  [Stanza parse]  →  [candidate generation]
  →  [E/S rule validation]  →  [BERT attention scoring]  →  [theory policy]
  →  [rerank]  →  predictions_enriched.jsonl + metrics.json + summary.json
```

## Input/Output Formats

**Input** (`bia_gold_sentences.jsonl`): JSONL with fields `sentence`, `doc_id`, `phrase_index`, and `gold` (list of `[arg1, rel, arg2]` triplets).

**Output per experiment** (in `--output-dir`):
- `*_predictions_enriched.jsonl/.csv` — all predicted triplets with linguistic metadata
- `*_metrics.json` — precision, recall, F1 with TP/FP/FN counts
- `*_summary.json` — top-level summary
- `*_selected_heads.json` — which BERT heads were used
- `*_per_pattern_metrics.csv` — breakdown by UD dependency pattern
- `*_per_rule_summary.csv` — effectiveness of each E/S rule
- `*_per_factuality_summary.csv` — breakdown by factuality (asserted/conditional/reported)
- `*_triples_table.csv` — gold vs. predicted side-by-side

## Ablation Study Structure

Results live in `ablation_experiments/`. Each experiment has its own subdirectory named `abl_rq<N>_<description>`. All 29 experiment summaries are aggregated in:
- `ablation_experiments/ablation_results.json`
- `ablation_experiments/ablation_results.csv/.xlsx`

Research questions covered: RQ1 (UD baseline), RQ2 (copula modes), RQ3 (theory rules), RQ4 (attention settings), RQ5 (span policy), RQ6 (individual rule ablation).

## Notable Implementation Details

- **Portuguese contractions** (`do`, `no`, `pelo`, etc.) are expanded to canonical form before canonicalization
- **Metalinguistic filters**: glosses like "em latim", "em inglês" are excluded as extraction arguments
- **Copula verbs** recognized: `ser`, `estar`, `ficar`, `parecer`, `tornar-se`, `virar`, `permanecer`, `continuar`
- **Token-to-wordpiece mapping** uses a heuristic alignment (`_map_tokens_to_wp_simple`) since Stanza and BERT tokenizations differ
- `bosque_heads_gain_roleaware()` ranks all 144 BERT heads (12 layers × 12 heads) by their ability to predict subject vs. object roles in the Bosque UD treebank — this runs at startup unless `--no-attn` is set
- Theory rules are individually toggleable via Config flags (e.g., `apply_e1`, `apply_s2`, etc.)
