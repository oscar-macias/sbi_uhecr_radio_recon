# Summary

This release delivers a compact, reproducible codebase for UHECR radio direction reconstruction using simulation-based inference with normalizing-flow posteriors. It adds a temperature scaling wrapper for calibrated posteriors, training/diagnostics utilities, and example notebooks with mock data. See the companion article (arXiv:2508.xxxxx) for details.

# Installation

## 1.  Create & activate the conda environment
```bash
conda create -n ili-torch python=3.10
conda activate ili-torch
```
## 2. Install the pipeline and all dependencies
### make sure pip is recent
pip install --upgrade pip

pip install -e "git+https://github.com/oscar-macias/sbi_uhecr_radio_recon.git#egg=sbi-uhecr-radio-recon"

## 3. How to cite this work?
Please cite the companion article (arXiv:2508.xxxxx / PRD DOI when available). A CITATION.cff is included.