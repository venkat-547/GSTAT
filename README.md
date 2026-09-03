# G-STAT: A Heterogeneous Graph Attention Framework for Genomic Variant Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-red.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Title:** G-STAT: A Heterogeneous Graph Attention Framework for Explainable Genomic Variant Triage
**Authors:** R. Venkat, K. Swathi, G. Srikanth
**Affiliation:** Department of Information Technology, Chaitanya Bharathi Institute of Technology (A), Hyderabad

---

## Overview

G-STAT is a heterogeneous Graph Neural Network for classifying genomic variants as **Pathogenic** or **Benign**, built directly on ClinVar. Rather than treating variants as independent rows of tabular data, G-STAT represents the problem as a **variant–gene graph**, where each variant node is connected to the gene it affects. This lets the model borrow signal from the surrounding gene neighborhood — clinical priors, constraint metrics, and co-occurring variant evidence — instead of relying purely on a variant's own features.

Core architectural ideas:

- **Type-specific gated projections** for variant and gene nodes (a GLU-style gate that learns which raw features are informative before any message passing occurs).
- **Dual-pathway bidirectional message passing** — a `GATv2` attention pathway for high-precision signal extraction and a `GraphSAGE` mean-aggregation pathway for robustness/imputation under missing data — fused additively at every layer.
- **Jumping Knowledge (max-pooling)** across layers to preserve useful low-order and high-order topological information while suppressing noise.
- **Focal Loss** to handle the heavy class imbalance inherent to clinical variant data.

### Benchmark Results (ClinVar, 80/20 split)

| Model                 | Accuracy   | Precision | Recall     | F1        |
|------------------------|:----------:|:---------:|:----------:|:---------:|
| XGBoost                | 98.07%     | 0.925     | 97.29%     | 0.948     |
| GCN (Homogeneous)      | 79.71%     | 0.371     | 14.71%     | 0.210     |
| GraphSAGE               | 97.46%     | 0.894     | 97.77%     | 0.934     |
| **G-STAT (Proposed)**  | **96.36%** | **0.844** | **98.29%** | **0.908** |

> G-STAT is optimized for **recall on the missense subset** — the clinically ambiguous variants where triage decisions matter most — rather than raw accuracy alone. See [Fair vs. Unfair Benchmarking](#fair-vs-unfair-benchmarking) below for why this distinction matters.

---

## Repository Structure

```
GSTAT/
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   └── s0_download_vcf.py          # Step 1: Download & decompress ClinVar VCF
│
├── hpc/
│   └── galaxy_hpc_instructions.md  # Step 2: SnpEff + REVEL annotation on Galaxy
│
├── src/
│   ├── gstat_architecture.py       # G-STAT model definition
│   ├── s01_vcf.py                  # Alternate VCF acquisition (GRCh38)
│   ├── s02_purist_tensor_compiler.py  # Step 3: Build heterogeneous tensors
│   ├── s03_index_builder.py        # Step 4: SQLite search index
│   ├── s04_train_model.py          # Step 5: Train G-STAT
│   ├── s05_app.py                  # Step 6: Interactive clinical dashboard
│   ├── s06_achitectural_benchmark.py  # G-STAT vs. GCN / GraphSAGE / XGBoost
│   ├── s07_live_validation.py      # Live validation against Ensembl REST API
│   ├── s08_unfair_SOTA_benchmark.py   # G-STAT vs. REVEL / CADD (full cohort)
│   ├── s09_fair_SOTA_benchmark.py  # G-STAT vs. REVEL / CADD (missense-only, bootstrapped CI)
│   └── s10_dna.py                  # Nucleotide-context extraction (FASTA)
│
├── output/                         # Generated metrics, figures, and model weights
├── processed/                      # Cached tensors (features, labels, edge indices)
│
└── docs/
    └── paper.pdf
```

---

## Installation

```bash
git clone https://github.com/<your-username>/GSTAT.git
cd GSTAT
pip install -r requirements.txt
```

A GPU with **≥6 GB VRAM** is recommended for training and inference; the pipeline falls back to CPU automatically if CUDA is unavailable.

---

## End-to-End Pipeline

### Step 1 — Acquire the ClinVar VCF

```bash
python data/s0_download_vcf.py
```

Streams and decompresses the official NCBI ClinVar VCF (~1.76 GB, GRCh37).

### Step 2 — Annotate on Galaxy HPC

Follow [`hpc/galaxy_hpc_instructions.md`](hpc/galaxy_hpc_instructions.md) to run **SnpEff** (functional consequence) and **SnpSift/dbNSFP** (REVEL and CADD scoring) on [usegalaxy.org](https://usegalaxy.org).

Output: `clinvar_enriched.vcf` (~6.7 GB), placed at:

```
E:\Genomic\data\vcf\clinvar_enriched.vcf
```

> On Linux/macOS, update the `BASE_DIR` constant at the top of each script accordingly.

### Step 3 — Compile Heterogeneous Tensors

```bash
python src/s02_purist_tensor_compiler.py
```

Parses the enriched VCF into variant/gene node features (15-D and 2-D respectively), integrates REVEL scores by genomic coordinate, assigns high-confidence pathogenic/benign labels from ClinVar review status, and writes `variant_metadata.csv`.

### Step 4 — Build the Search Index

```bash
python src/s03_index_builder.py
```

Compiles `variant_metadata.csv` into an indexed SQLite database for fast lookup by RSID, ClinVar ID, or coordinate.

### Step 5 — Train G-STAT

```bash
python src/s04_train_model.py
```

Trains the heterogeneous GNN using Focal Loss with a `NeighborLoader`-based mini-batch sampler, tracking missense-specific F1 for checkpointing.

### Step 6 — Benchmark & Validate

```bash
python src/s06_achitectural_benchmark.py   # vs. GCN, GraphSAGE, XGBoost
python src/s08_unfair_SOTA_benchmark.py    # vs. REVEL, CADD (full cohort)
python src/s09_fair_SOTA_benchmark.py      # vs. REVEL, CADD (missense-only, bootstrapped CIs)
python src/s07_live_validation.py          # Live concordance check against Ensembl
```

### Step 7 — Launch the Clinical Dashboard

```bash
python src/s05_app.py
```

Serves an interactive Dash application at `http://127.0.0.1:8050` for querying individual variants, visualizing the local variant–gene neighborhood, and inspecting per-feature saliency (occlusion-based explainability).

### Optional — Nucleotide Context Extraction

```bash
python src/s10_dna.py
```

Pulls surrounding reference/alternate DNA sequence context per variant via `pyfaidx`, for downstream sequence-level analysis.

---

## Fair vs. Unfair Benchmarking

Two benchmark scripts are included deliberately:

- **`s08_unfair_SOTA_benchmark.py`** evaluates G-STAT against REVEL and CADD across the *entire* labeled cohort, including variant classes (nonsense, frameshift, synonymous) that REVEL was never designed to score.
- **`s09_fair_SOTA_benchmark.py`** restricts evaluation to the **missense-only** subset — REVEL and CADD's intended domain — and reports **95% bootstrapped confidence intervals** for a statistically defensible comparison.

Reporting only the first would overstate G-STAT's advantage; the second is the honest, apples-to-apples comparison.

---

## Feature Space

Each variant node is represented by a 15-dimensional feature vector spanning:

| Category | Features |
|---|---|
| Sequence Ontology | Missense / Synonymous / Truncating / Unknown consequence |
| Structural Impact | SnpEff HIGH / MODERATE / LOW / UNKNOWN impact flags |
| Population Genetics | ExAC allele frequency (−log₁₀ transformed), AF-tracked flag |
| Clinical Priors | ACMG-59 actionable gene flag, high-pLI (LoF-intolerant) gene flag |
| Review Confidence | ClinVar multi-submitter / single-submitter review status |
| Thermodynamics | REVEL pathogenicity score |

Gene nodes carry a 2-dimensional feature vector (ACMG status, pLI-based fragility).

---

## Data Sources

- **ClinVar** — [ftp.ncbi.nlm.nih.gov/pub/clinvar](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/)
- **SnpEff** — functional effect annotation
- **dbNSFP / REVEL** — variant pathogenicity scoring
- **Ensembl REST API** — independent clinical concordance validation

---

## Citation

```bibtex
@article{venkat2025gstat,
  title       = {G-STAT: A Heterogeneous Graph Attention Framework for
                 Explainable Genomic Variant Triage},
  author      = {Venkat, R. and Swathi, K. and Srikanth, G.},
  journal     = {Journal Name},
  year        = {2025},
  institution = {Chaitanya Bharathi Institute of Technology (A), Hyderabad}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**R. Venkat**
Email: rayudujeevan3@gmail.com
Department of Information Technology, CBIT (A), Hyderabad
