# Installation

## 1.  Create & activate the research environment
```bash
conda create -n ili-torch python=3.10
conda activate ili-torch
```
## 2. Install the pipeline and all dependencies
### make sure pip is recent
pip install --upgrade pip

pip install -e "git+https://github.com/<your-org>/sbi_uhecr_radio_recon.git#egg=sbi-uhecr-radio-recon"
