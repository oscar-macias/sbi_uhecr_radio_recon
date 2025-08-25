# Summary

This release delivers a compact, reproducible codebase for UHECR radio direction reconstruction using simulation-based inference with normalizing-flow posteriors. It adds a temperature scaling wrapper for calibrated posteriors, training/diagnostics utilities, and example notebooks with mock data. See the companion article (arXiv:2508.15991) for details.

# Installation

## 1.  Create & activate the conda environment
```bash
conda create -n ili-torch python=3.10
conda activate ili-torch
```
## 2. Install the pipeline and all dependencies
> make sure pip is recent

```bash
pip install --upgrade pip

pip install -e "git+https://github.com/oscar-macias/sbi_uhecr_radio_recon.git#egg=sbi-uhecr-radio-recon"
```

## 3. How to cite this work?
If you use this package, please cite the companion article:
- **Macias et al. (2025), https://arxiv.org/abs/2508.15991**

In addition, please consider citing the packages on which this work builds:

- Ho et al. (2024), ``LtU-ILI: An All-in-One Framework for Implicit Inference in Astrophysics and Cosmology'', https://arxiv.org/abs/2402.05137

- Ferriere et al. (2024), ``Analytical planar wavefront reconstruction and error estimates for radio detection of extensive air showers'', https://arxiv.org/abs/2405.09377
