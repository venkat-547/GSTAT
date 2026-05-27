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

G-STAT is a heterogeneous Graph Neural Network designed to classify genomic variants as Pathogenic or Benign. The model processes variant and gene features separately using type-specific projections, bi-directional GATv2 message passing, Jumping Knowledge, and Focal Loss.

Performance on ClinVar dataset (80/20 split):

| Model                | Accuracy | Precision | Recall   | F1    |
|----------------------|----------|-----------|----------|-------|
| XGBoost              | 98.07%   | 0.925     | 97.29%   | 0.948 |
| GCN (Homogeneous)    | 79.71%   | 0.371     | 14.71%   | 0.210 |
| GraphSAGE            | 97.46%   | 0.894     | 97.77%   | 0.934 |
| **G-STAT (Proposed)**| **96.36%** | **0.844** | **98.29%** | **0.908** |

---

## Repository Structure
GSTAT/
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   └── step1_download_vcf.py
│
├── hpc/
│   └── galaxy_hpc_instructions.md
│
├── src/
│   ├── gstat_architecture.py
│   ├── s02_purist_tensor_compiler.py
│   ├── s03_train_model.py
│   ├── s03_1_evaluation.py
│   ├── comparison.py
│   ├── s04_index_builder.py
│   ├── s05_final.py
│   └── ...
│
├── audit/
│   └── ...
│
└── docs/
└── paper.pdf
text---

## End-to-End Pipeline

### Prerequisites

```bash
pip install torch torch-geometric xgboost scikit-learn pandas numpy \
            requests plotly dash networkx matplotlib
A GPU with ≥6 GB VRAM is recommended for training.

STEP 1 — Download ClinVar VCF
Bashcd data
python step1_download_vcf.py
Downloads and extracts the official ClinVar GRCh37 VCF file (~1.76 GB).

STEP 2 — Enrich VCF on Galaxy HPC
Follow the detailed instructions in hpc/galaxy_hpc_instructions.md to annotate the VCF using SnpEff and REVEL on Galaxy.
Output: clinvar_enriched.vcf (~6.7 GB)
Place the file at:
textE:\Genomic\data\vcf\clinvar_enriched.vcf

STEP 3 — Compile Heterogeneous Tensors
Bashpython src/s02_purist_tensor_compiler.py

STEP 4 — Train G-STAT
Bashpython src/s03_train_model.py

STEP 5 — Benchmark & Evaluation
Bashpython src/comparison.py
python src/s03_1_evaluation.py

STEP 6 — Build Search Index
Bashpython src/s04_index_builder.py

STEP 7 — Launch Clinical Dashboard
Bashpython src/s05_final.py
Access the interface at http://127.0.0.1:8050

Data Sources

ClinVar: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/
SnpEff: Functional effect annotation
dbNSFP / REVEL: Pathogenicity scoring


Citation
bibtex@article{venkat2025gstat,
  title   = {G-STAT: A Heterogeneous Graph Attention Framework for 
             Explainable Genomic Variant Triage},
  author  = {Venkat, R. and Swathi, K. and Srikanth, G.},
  journal = {Journal Name},
  year    = {2025},
  institution = {Chaitanya Bharathi Institute of Technology (A), Hyderabad}
}

License
This project is licensed under the MIT License. See LICENSE for details.

Contact
R. Venkat
Email: rayudujeevan3@gmail.com
Department of Information Technology, CBIT(A), Hyderabad
text---

You can now copy the entire content above and save it as `README.md`.

Would you like me to generate the `galaxy_hpc_instructions.md` file next? Just say yes.