# DPC-OpenNet

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This repository contains the implementation of **DPC-OpenNet: Dual-view
Prototype-Contrastive Network for Open-world Fault Diagnosis**, accompanying
*Few-shot multi-unknown fault diagnosis of power-shift system in continuously
variable transmission tractors*.

## Repository Contents

```text
models/dpc_opennet.py    
Experiment/            
compare/                
data/README.md          
tools/                  
tests/                  
run_experiments.py      
predict.py              
requirements.txt       
```

## Installation

Use Python 3.10 and install the dependencies:

```bash
python -m pip install -r requirements.txt
python tools/check_environment.py
```

For CUDA 11.8, install the matching PyTorch wheel before the other dependencies:

```bash
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

## Dataset

The author do not have permission to share data.

## Training and Testing

Run a single 20-shot F2-unknown experiment for 100 epochs:

```bash
python Experiment/20shot_F2_unknown.py --dataset-root /path/to/dataset_8000
```

Run the proposed method with ten independent seeds at all shot levels and
unknown configurations:

```bash
python run_experiments.py --method proposed --dataset-root /path/to/dataset_8000
```

Run the comparison suite, or inspect the complete plan without training:

```bash
python run_experiments.py --method baselines --dataset-root /path/to/dataset_8000
python run_experiments.py --method all --dry-run
```


## Inference

```bash
python predict.py --checkpoint path/to/final_model.pt --input path/to/csv_samples --output results/predictions.csv
```

## License and Usage

All resources in this repository are for academic peer review only at present. Prior to the official publication of the corresponding paper, unauthorized reproduction, modification, commercial use, public distribution and secondary creation of any part of this project are strictly prohibited.

Once the paper is accepted and formally published, all resources will be fully open-sourced under the MIT License. Researchers may use, reference and modify the code and datasets for non-commercial academic research, provided that the original paper is properly cited.

The author retains all intellectual property rights of this project. Any unauthorized academic appropriation, plagiarism or act of claiming research priority will be held accountable in accordance with academic norms and relevant intellectual property laws and regulations.

## Disclaimer

This is an unpublished academic work. All contents in this repository are temporary review materials and do not represent the final published version. Contents may be adjusted, optimized and updated during the peer review process. The author shall not be liable for any unauthorized use or misinterpretation of the current resources by third parties.
