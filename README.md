<div align="center">

# 🧠 LLM Drift & Hallucination Detection System
### *Multi-Model · Multi-Approach · Production-Grade Evaluation Pipeline*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

<br/>

> A comprehensive, multi-approach pipeline for detecting **hallucinations** and **response drift** in large language models.  
> Evaluates **Gemini**, **Perplexity**, and **UltraChat** datasets using three progressively sophisticated detection strategies — from TF-IDF to SOTA transformer ensembles.

<br/>

```
┌─────────────────────────────────────────────────────────┐
│   LLM Response   →   3-Approach Detection Pipeline   →  Hallucination Score + Model Ranking  │
└─────────────────────────────────────────────────────────┘
```

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Datasets](#-datasets)
- [System Architecture](#-system-architecture)
- [Approach 1 — TF-IDF + Linguistic Features](#-approach-1--tfidf--linguistic-features)
- [Approach 2 — MPNet + DeBERTa NLI](#-approach-2--mpnet--deberta-nli)
- [Approach 3 — SBERT + MMR Retrieval + SOTA Ensemble](#-approach-3--sbert--mmr-retrieval--sota-ensemble)
- [Hallucination Scoring Formula](#-hallucination-scoring-formula)
- [Model Rankings & Output](#-model-rankings--output)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)

---

## 🔬 Overview

This project builds a **multi-layered hallucination and drift detection system** for LLM-generated responses. It processes real conversation datasets from leading AI models, extracts user-assistant pairs, and evaluates how faithfully and accurately each model responds to user prompts.

### What is Hallucination in LLMs?

| Type | Description |
|------|-------------|
| **Intrinsic Hallucination** | Response contradicts the source/context |
| **Extrinsic Hallucination** | Response adds information unverifiable from context |
| **Drift** | Responses deviate statistically from a model's expected behaviour distribution |

### Key Capabilities

- ✅ **Multi-dataset support** — Gemini, Perplexity, UltraChat conversations
- ✅ **Three detection approaches** — lightweight to SOTA, switchable per use case
- ✅ **Semantic + NLI + Cross-Encoder ensemble** for robust scoring
- ✅ **BM25 + Semantic Hybrid Retrieval** with MMR diversity reranking
- ✅ **Faithfulness, Context Relevance, Citation Accuracy, Answer Relevance** metrics
- ✅ **GPU-accelerated** batch inference with checkpoint/resume support
- ✅ **Statistical significance testing** via one-way ANOVA
- ✅ **PDF report generation** with ReportLab

---

## 📦 Datasets

Three LLM conversation datasets are processed and merged into a unified pipeline:

| Dataset | Source | Description |
|---------|--------|-------------|
| **Gemini Sample** | `gemini-sample.json` | Multi-turn conversations from Google Gemini |
| **Perplexity Sample** | `perplexity-sample.json` | Conversations from Perplexity AI (with JSON malformation handling) |
| **UltraChat** | `ultrachat_sample.csv` | Large-scale instructional dialogue dataset (80/20 train-test split) |

### Preprocessing Pipeline

```python
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # Remove URLs
    text = re.sub(r"[#*_>`~\-]+", " ", text)         # Strip markdown
    text = re.sub(r"[^a-z\s]", " ", text)            # Keep only alpha
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

> **Note:** The Perplexity dataset contains a structural JSON bug (`"role": "assistant":` instead of `"role": "assistant", "message":`). The pipeline handles this automatically via regex-based repair before parsing.

---

## 🏗 System Architecture

```
Raw JSON / CSV Datasets
        │
        ▼
┌──────────────────────┐
│  Data Loading &      │
│  Text Preprocessing  │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  User-Assistant      │
│  Pair Creation       │  ← pivot_table on (model_name, id, role)
└──────────────────────┘
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
┌────────────────┐                     ┌─────────────────────┐
│  Approach 1    │                     │  Approach 2 & 3     │
│  TF-IDF +      │                     │  Transformer-Based  │
│  Linguistic    │                     │  NLI + Retrieval    │
└────────────────┘                     └─────────────────────┘
        │                                         │
        └──────────────────┬──────────────────────┘
                           ▼
              ┌────────────────────────┐
              │  Hallucination Score   │
              │  (Weighted Composite)  │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Model Rankings +      │
              │  ANOVA Significance    │
              │  + PDF Report          │
              └────────────────────────┘
```

---

## ⚡ Approach 1 — TF-IDF + Linguistic Features

**Speed tier: `fast` / `ultra_fast`** — Lightweight, no GPU required.

### Features Extracted

| Feature | Description | Weight |
|---------|-------------|--------|
| `tfidf_similarity` | Cosine similarity between user prompt & assistant response (TF-IDF, 5000 features, bigrams) | 0.30 |
| `length_ratio` | Ratio of assistant response length to user prompt length | 0.15 |
| `vocab_overlap` | Fraction of user vocabulary present in the response | 0.15 |
| `repetition_score` | Bigram repetition rate — detects degenerate responses | 0.15 |
| `hedging_score` | Density of uncertainty markers (maybe, perhaps, unclear…) | 0.10 |
| `similarity_deviation` | Z-score deviation from model's own similarity distribution | 0.10 |
| `lexical_diversity` | Unique word ratio in assistant response | 0.05 |

### Speed Tiers

```python
SPEED_TIER = 'ultra_fast'   # TF-IDF only — fastest
SPEED_TIER = 'fast'         # TF-IDF + Linguistic — recommended
SPEED_TIER = 'balanced'     # All features + Transformer embeddings
```

---

## 🤖 Approach 2 — MPNet + DeBERTa NLI

**State-of-the-art NLI-based detection.** Combines semantic embeddings with natural language inference to detect logical contradictions between prompts and responses.

### Models Used

| Model | Purpose | Notes |
|-------|---------|-------|
| `paraphrase-mpnet-base-v2` | Semantic similarity (bi-encoder) | 768-dim embeddings, normalized cosine |
| `microsoft/deberta-v3-large` | NLI entailment / contradiction | 3-class: entailment · neutral · contradiction |

### NLI Faithfulness Score

```python
# High entailment = faithful response
# High contradiction = hallucination signal
pairs_df['nli_faithfulness'] = pairs_df['nli_entailment'] - pairs_df['nli_contradiction']
```

### Hallucination Score Weights (Approach 2)

| Component | Source | Weight |
|-----------|--------|--------|
| NLI Contradiction | DeBERTa-v3 | **0.35** |
| Semantic Dissimilarity | MPNet | 0.25 |
| Ungrounded Vocabulary | Overlap metric | 0.15 |
| Uncertainty Language | Hedging score | 0.10 |
| Repetition | Bigram degeneration | 0.05 |
| Semantic Anomaly | Z-score deviation | 0.05 |
| Length Anomaly | Z-score deviation | 0.05 |

---

## 🔍 Approach 3 — SBERT + MMR Retrieval + SOTA Ensemble

**The most comprehensive approach.** Combines semantic retrieval, cross-encoder reranking, NLI verification, and a mock knowledge graph into a 4-stage ensemble.

### Retrieval System

Three retrieval strategies are implemented and compared:

```
Semantic Retrieval (SBERT cosine)
         +
BM25 Sparse Retrieval
         ↓
Reciprocal Rank Fusion (RRF) → Hybrid Retrieval
         ↓
MMR Reranking (λ=0.7)        → Diverse, Relevant Context
```

**MMR (Maximal Marginal Relevance)** balances relevance vs. redundancy:

```python
MMR(doc) = λ × relevance(doc, query) − (1 − λ) × max_similarity(doc, selected)
```

### Evaluation Metrics (Approach 3)

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Fraction of response claims supported by retrieved context |
| **Context Relevance** | Mean cosine similarity between query and retrieved contexts |
| **Citation Accuracy** | Fraction of citations in response verified against source documents |
| **Answer Relevance** | Cosine similarity between query and full response |
| **Hallucination Index** | Weighted composite (0–100, lower is better) |

```python
hallucination_index = (
    (1 - faithfulness)      × 0.35 +
    (1 - context_relevance) × 0.25 +
    (1 - citation_accuracy) × 0.25 +
    (1 - answer_relevance)  × 0.15
) × 100
```

### SOTA 4-Stage Detector

```
┌─────────────────────────────────────────────────────────────┐
│              StateOfTheArtHallucinationDetector              │
├──────────────┬───────────────┬────────────────┬─────────────┤
│  Stage 1     │   Stage 2     │   Stage 3      │  Stage 4    │
│  Semantic    │   NLI Verify  │  Cross-Encoder │  Knowledge  │
│  Similarity  │  (BART-MNLI)  │  Precision     │  Graph Check│
│  (MPNet)     │               │  (MS-MARCO)    │             │
│  Weight: 25% │  Weight: 35%  │  Weight: 25%   │  Weight:15% │
└──────────────┴───────────────┴────────────────┴─────────────┘
                           ↓
              Weighted Ensemble Final Score
              is_hallucination: score < 0.5
```

**Models loaded in SOTA detector:**

| Component | Model |
|-----------|-------|
| Bi-encoder embeddings | `all-mpnet-base-v2` |
| NLI classification | `facebook/bart-large-mnli` |
| Cross-encoder relevance | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Factual claim detection | Regex pattern-based (dates, %, named entities) |

### GPU Acceleration

```python
detector = StateOfTheArtHallucinationDetector(use_gpu=True)
# Batch size: 32 on GPU | 8 on CPU
# Checkpoint every 1000 samples — resumes automatically
# torch.cuda.empty_cache() called after each checkpoint batch
```

---

## 📊 Hallucination Scoring Formula

All three approaches normalize to a **0–1 hallucination score** (higher = more hallucinated):

```
Approach 1 (Fast):
  score = 0.30×(1−tfidf_sim) + 0.15×length_ratio + 0.15×(1−vocab_overlap)
        + 0.15×repetition + 0.10×hedging + 0.10×sim_deviation + 0.05×(1−lex_diversity)

Approach 2 (NLI):
  score = 0.35×nli_contradiction + 0.25×(1−semantic_sim) + 0.15×(1−vocab_overlap)
        + 0.10×hedging + 0.05×repetition + 0.05×sem_anomaly + 0.05×len_anomaly

Approach 3 (SOTA Ensemble):
  score = 0.25×(1−semantic) + 0.35×(1−nli) + 0.25×(1−cross_encoder) + 0.15×(1−kg)
```

**Quality Thresholds:**

| Score | Category |
|-------|----------|
| < 0.25 (< 25%) | 🟢 Excellent |
| 0.25 – 0.50 | 🟡 Good |
| 0.50 – 0.75 | 🟠 Fair |
| > 0.75 (> 75%) | 🔴 Poor |

---

## 🏆 Model Rankings & Output

The pipeline produces **per-model rankings** with ANOVA statistical significance testing:

```
MODEL RANKINGS (Best to Worst — lower hallucination score wins)
─────────────────────────────────────────────
  1. gemini          Score: X.XXXX (XX.XX%) — Excellent
  2. ultrachat       Score: X.XXXX (XX.XX%) — Good
  3. perplexity      Score: X.XXXX (XX.XX%) — Good

ANOVA F-statistic: XX.XXXX
Significant difference between models: YES / NO
```

### Output Files

| File | Description |
|------|-------------|
| `hallucination_scores.csv` | Per-pair scores from Approach 1 |
| `model_comparison.csv` | Aggregated model-level statistics |
| `hallucination_analysis.csv` | Full feature matrix from Approach 2 |
| `hallucination_evaluation_results.csv` | Faithfulness, context relevance, citation accuracy (Approach 3) |
| `sota_hallucination_results_final.csv` | SOTA 4-stage ensemble results |
| `sota_checkpoint_N.csv` | Incremental checkpoints for large runs |

---

## 📁 Project Structure

```
llm-drift-hallucination-detection/
│
├── LLM_Drift_and_Hallucination_Detection_System.ipynb   # Main notebook
│
├── data/
│   ├── gemini-sample.json              # Gemini conversation data
│   ├── perplexity-sample.json          # Perplexity conversation data (raw)
│   ├── perplexity-sample-fixed.json    # Auto-repaired Perplexity JSON
│   ├── ultrachat_sample.csv            # UltraChat dataset
│   ├── gemini-sample-updated.csv       # Cleaned Gemini CSV
│   ├── perplexity-sample-updated.csv   # Cleaned Perplexity CSV
│   └── combined_dataset.csv            # Merged dataset (all models)
│
├── outputs/
│   ├── hallucination_scores.csv
│   ├── model_comparison.csv
│   ├── hallucination_analysis.csv
│   ├── hallucination_evaluation_results.csv
│   └── sota_hallucination_results_final.csv
│
└── README.md
```

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install sentence-transformers transformers torch pandas numpy scipy \
            scikit-learn matplotlib seaborn reportlab accelerate rank_bm25 \
            nltk tqdm
```

### Run on Google Colab (Recommended)

1. Upload the notebook to [Google Colab](https://colab.research.google.com/)
2. Enable GPU runtime: **Runtime → Change runtime type → T4 GPU**
3. Upload your dataset files to `/content/`
4. Run cells in order — each approach is self-contained

### Run Locally

```bash
git clone https://github.com/your-username/llm-drift-hallucination-detection.git
cd llm-drift-hallucination-detection
jupyter notebook LLM_Drift_and_Hallucination_Detection_System.ipynb
```

### Choose Your Approach

| Use Case | Approach | Time | GPU Required |
|----------|----------|------|--------------|
| Quick benchmark, large dataset | Approach 1 (`fast`) | Minutes | ❌ |
| Balanced accuracy vs. speed | Approach 2 (MPNet + DeBERTa) | ~1–2 hrs | ✅ Recommended |
| Maximum accuracy, production eval | Approach 3 (SOTA Ensemble) | Hours | ✅ Required |

---

## 🛠 Tech Stack

| Tool | Role |
|------|------|
| ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=flat) | Deep learning backend |
| ![HuggingFace](https://img.shields.io/badge/-HuggingFace-FFD21E?logo=huggingface&logoColor=black&style=flat) | Transformer models (DeBERTa, BART-MNLI) |
| ![SentenceTransformers](https://img.shields.io/badge/-SentenceTransformers-4B8BBE?style=flat) | MPNet, MiniLM bi-encoders, cross-encoders |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat) | TF-IDF, cosine similarity, PCA |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat) | Numerical computation |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat) | Dataset manipulation |
| ![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?logo=scipy&logoColor=white&style=flat) | Wasserstein distance, ANOVA |
| ![NLTK](https://img.shields.io/badge/-NLTK-154F5B?style=flat) | Sentence tokenization, stopwords |
| ![rank_bm25](https://img.shields.io/badge/-BM25-4A90D9?style=flat) | Sparse retrieval |
| ![ReportLab](https://img.shields.io/badge/-ReportLab-CC0000?style=flat) | PDF report generation |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat) | Visualisation |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
