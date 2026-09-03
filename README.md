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

G-STAT is a heterogeneous Graph Neural Network for classifying genomic variants as **Pathogenic** or **Benign**, built directly on ClinVar. Instead of treating variants as independent rows of tabular data, G-STAT represents the problem as a **variant–gene graph**, where each variant node is connected to the gene it affects. This lets the model borrow signal from the surrounding gene neighborhood — clinical priors, constraint metrics, and co-occurring variant evidence — instead of relying purely on a variant's own features.

Core architectural ideas:

- **Type-specific gated projections** for variant and gene nodes — a GLU-style gate that learns which raw features are informative before any message passing occurs.
- **Dual-pathway bidirectional message passing** — a `GATv2` attention pathway for high-precision signal extraction, fused additively with a `GraphSAGE` mean-aggregation pathway for robustness under missing or noisy data.
- **Jumping Knowledge (max-pooling)** across layers to preserve useful low-order and high-order topological information while suppressing noise.
- **Focal Loss** to handle the heavy class imbalance inherent to clinical variant data.
- **Occlusion-based explainability** (feature-level and pathway-level saliency) surfaced directly in the interactive dashboard.

### Benchmark Results (ClinVar, 80/20 split)

| Model                  | Accuracy   | Precision | Recall     | F1        |
|-------------------------|:----------:|:---------:|:----------:|:---------:|
| XGBoost                 | 98.07%     | 0.925     | 97.29%     | 0.948     |
| GCN (Homogeneous)       | 79.71%     | 0.371     | 14.71%     | 0.210     |
| GraphSAGE                | 97.46%     | 0.894     | 97.77%     | 0.934     |
| **G-STAT (Proposed)**   | **96.36%** | **0.844** | **98.29%** | **0.908** |

> G-STAT is optimized for **recall on the missense subset** — the clinically ambiguous variants where triage decisions matter most — rather than raw accuracy alone. See [Fair vs. Unfair Benchmarking](#fair-vs-unfair-benchmarking) for why this distinction matters.

---

## Repository Structure

```
GSTAT/
├── README.md
├── requirements.txt
├── LICENSE
│
├── gstat_architecture.py            # Core G-STAT model definition (shared across all scripts)
│
├── s0_download_vcf.py               # Step 1: Download & decompress ClinVar VCF (GRCh37)
├── s01_vcf.py                       # Alternate VCF acquisition path (GRCh38)
│
├── galaxy_hpc_instructions.md       # Step 2: SnpEff + REVEL annotation via Galaxy HPC
├── galaxy_instructions.md           # Extended walkthrough of the same Galaxy pipeline
│
├── s02_purist_tensor_compiler.py    # Step 3: Build heterogeneous variant/gene tensors
├── s03_index_builder.py             # Step 4: Compile SQLite search index for fast lookup
├── s04_train_model.py               # Step 5: Train G-STAT with Focal Loss
│
├── s06_achitectural_benchmark.py    # G-STAT vs. GCN / GraphSAGE / XGBoost
├── s08_unfair_SOTA_benchmark.py     # G-STAT vs. REVEL / CADD — full cohort
├── s09_fair_SOTA_benchmark.py       # G-STAT vs. REVEL / CADD — missense-only, bootstrapped CIs
├── s07_live_validation.py           # Live concordance check against the Ensembl REST API
│
├── s05_app.py                       # Step 6: Interactive Dash clinical dashboard
├── s10_dna.py                       # Optional: DNA-level detail tensor (variant type, sequence change)
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

A GPU with **≥6 GB VRAM** is recommended for training and inference. The pipeline falls back to CPU automatically if CUDA is unavailable.

By default, all scripts point at a Windows-style base directory:

```python
BASE_DIR = r"E:\Genomic"
```

On Linux/macOS, update this constant at the top of each script to point at your working directory (e.g. `BASE_DIR = "/home/user/genomic"`).

---

## End-to-End Pipeline

### Step 1 — Acquire the ClinVar VCF

```bash
python s0_download_vcf.py
```

Streams and decompresses the official NCBI ClinVar VCF (~1.76 GB, GRCh37). `s01_vcf.py` provides an alternate path for the GRCh38 build if that's your reference genome of choice.

### Step 2 — Annotate on Galaxy HPC

Follow [`galaxy_hpc_instructions.md`](galaxy_hpc_instructions.md) to run **SnpEff** (functional consequence) and **SnpSift/dbNSFP** (REVEL and CADD scoring) on [usegalaxy.org](https://usegalaxy.org) — no local install required.

Output: `clinvar_enriched.vcf` (~6.7 GB), placed at:

```
E:\Genomic\data\vcf\clinvar_enriched.vcf
```

### Step 3 — Compile Heterogeneous Tensors

```bash
python s02_purist_tensor_compiler.py
```

Parses the enriched VCF into variant/gene node features (15-D and 2-D respectively), integrates REVEL scores by genomic coordinate, assigns high-confidence pathogenic/benign labels from ClinVar review status, and writes `variant_metadata.csv`.

### Step 4 — Build the Search Index

```bash
python s03_index_builder.py
```

Compiles `variant_metadata.csv` into a de-duplicated, indexed SQLite database for fast lookup by RSID, ClinVar ID, or genomic coordinate.

### Step 5 — Train G-STAT

```bash
python s04_train_model.py
```

Trains the heterogeneous GNN using Focal Loss with a `NeighborLoader`-based mini-batch sampler, tracking missense-specific F1 for model checkpointing.

### Step 6 — Benchmark & Validate

```bash
python s06_achitectural_benchmark.py   # vs. GCN, GraphSAGE, XGBoost
python s08_unfair_SOTA_benchmark.py    # vs. REVEL, CADD (full cohort)
python s09_fair_SOTA_benchmark.py      # vs. REVEL, CADD (missense-only, bootstrapped CIs)
python s07_live_validation.py          # Live concordance check against Ensembl
```

### Step 7 — Launch the Clinical Dashboard

```bash
python s05_app.py
```

Serves an interactive Dash application at `http://127.0.0.1:8050` for querying individual variants, visualizing the local variant–gene neighborhood, and inspecting per-feature saliency via occlusion-based explainability.

### Optional — DNA-Level Detail Extraction

```bash
python s10_dna.py
```

Parses the enriched VCF directly for per-variant nucleotide-level detail — variant type (SNV/INS/DEL/MNV), sequence change, and structural impact — used to enrich the dashboard's variant detail view.

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
