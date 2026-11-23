# DeepGene-AIID

**DeepGene-AIID** is a multimodal dual-channel graph neural network framework for prioritizing autoimmune and autoinflammatory disease-associated genes (AAGs) and dissecting shared pan-AIID functional modules.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Autoimmune and autoinflammatory diseases (AIIDs) are clinically heterogeneous yet frequently co-occur and share underlying molecular mechanisms. **DeepGene-AIID** integrates genomic (GWAS), transcriptomic (17 blood-derived GEO cohorts, $n = 1,852$), and network structural modalities across **11,058 protein-coding genes** to prioritize candidate AAGs.

The framework combines:
1. **Dual-Channel Graph Encoding**: Weighted GraphSAGE and Directed Graph Convolutional Networks (DiGCN) to capture undirected and directed biological network topologies.
2. **Adversarial Representation Disentanglement**: Separates shared cross-modal consensus features from modality-specific representations (genomic-specific, expression-specific, and structure-specific).
3. **SimAM-Guided Feature Fusion**: Parameter-free attention mechanism to adaptively weight disentangled representations for final gene prioritization.

---

## Framework Architecture

<p align="center">
  <img src="figures/Figure_1.png" alt="Overview of the DeepGene-AIID framework" width="90%">
</p>

*Figure 1. Overall architecture of DeepGene-AIID, illustrating multimodal data integration across 17 AIID entities, dual-channel graph encoding, adversarial feature disentanglement, SimAM-guided fusion, and downstream translational and modular analyses.*

---

## Key Highlights

- **Benchmark Performance**: Achieved mean **AUROC = 0.928** and **AUPR = 0.846** in 5-fold cross-validation on a curated benchmark of 660 known AAGs and 1,000 putative negative genes, outperforming baseline models and single-channel variants.
- **Clinical Generalizability**: Transcriptomic gene-set scores derived from prioritized genes correlated with clinical phenotypes in independent systemic lupus erythematosus (SLE, `GSE121239`) and primary Sjögren’s syndrome (pSS, `GSE84844`) cohorts.
- **Novel Candidate Discovery**: Prioritized **110 candidate AAGs** (score > 0.95) outside the positive training set, yielding **93 novel candidate AAGs** after excluding 17 overlapping pharmacological reference targets.
- **Pan-AIID Modular Landscape**: Integrated known AAGs, pharmacological targets, and novel candidates into an **807-gene network** partitioned into **7 functional modules** spanning canonical autoimmune pathways, genomic maintenance, metabolic reprogramming/protein N-glycosylation, and neuroendocrine/GPCR signaling.

---

## Repository Structure

```text
DeepGene-AIID/
├── data/                   # Preprocessed multimodal features, PPI/directed networks, and labels
├── figures/                # Manuscript figures (Figure_1.png)
├── models/                 # Dual-channel encoder, disentanglement, and SimAM modules
├── scripts/                # Training, 5-fold CV evaluation, and downstream analysis scripts
├── supplementary/          # Supplementary Tables 1 & 2
├── results/                # Prioritized gene scores and module assignments
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md
```

---

## Installation

We recommend creating a dedicated Conda environment:

```bash
git clone https://github.com/<your-username>/DeepGene-AIID.git
cd DeepGene-AIID

conda create -n deepgene_aiid python=3.9
conda activate deepgene_aiid

pip install -r requirements.txt
```

### Core Dependencies
- `Python >= 3.9`
- `PyTorch >= 2.0`
- `PyTorch Geometric`
- `numpy`, `pandas`, `scikit-learn`, `networkx`

---

## Usage

### 1. 5-Fold Cross-Validation
```bash
python scripts/train_cv.py --config configs/default.yaml
```

### 2. Genome-Wide Prioritization (11,058 Protein-Coding Genes)
```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pt --output results/prioritized_genes.csv
```

---

## Data Availability

- **Training Cohorts**: Multimodal GWAS and transcriptomic metadata for the 17 AIID entities are detailed in `supplementary/Supplementary_Table_1.xlsx`.
- **Gene Lists & Annotations**: Predicted AAGs (110), known AAGs (660), small-molecule targets (109), novel candidate AAGs (93), core network genes (844), and antibody/biologic targets (64) are provided in `supplementary/Supplementary_Table_2.xlsx`.
- **External Validation Cohorts**: Publicly available via NCBI GEO under accessions `GSE121239` (SLE) and `GSE84844` (pSS).

---



## Contact

For questions or issues, please open a GitHub Issue or contact the corresponding author at `<liyuanyuan@sibpt.cn>`.
