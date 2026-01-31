# DifMTT: Towards Multi-Treatment Task Learning and Subclass Balancing for Medication Recommendation

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red)](https://pytorch.org/)

This repository provides the official PyTorch implementation for the paper:

> **Towards Multi-Treatment Task Learning and Subclass Balancing for Medication Recommendation**


## Key Features

- **Subclass-Balancing Contrastive Learning (SBCL)**: Addresses class imbalance through bi-granularity contrastive learning with dynamic temperature adjustment
- **Diffusion-based Generation**: Leverages Gaussian diffusion process for robust medication set generation with multi-modal distribution modeling
- **DDI-aware Guidance**: Integrates drug-drug interaction awareness during the diffusion sampling process
- **Multi-Treatment Task Learning**: Captures diverse patient treatment intents for personalized recommendations
- **Molecular Substructure Encoding**: Uses GNN-based molecular representation learning for drug embeddings

## Installation

1. Clone this git repository and change directory to this repository:

   ```shell
   git clone https://github.com/your-repo/DifMTT.git
   cd DifMTT/
   ```

2. A new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) is suggested. 

   ```bash
   conda create --name DifMTT python=3.8
   ```

3. Activate the newly created environment.

   ```bash
   conda activate DifMTT
   ```



## Download the data

1. You must have obtained access to [MIMIC-III](https://physionet.org/content/mimiciii/) and [MIMIC-IV](https://physionet.org/content/mimiciv/) databases before running the code. 

2. Download the MIMIC-III and MIMIC-IV datasets, then unzip and put them in the `data/input/` directory. Specifically, you need to download the following files from MIMIC-III: `DIAGNOSES_ICD.csv`, `PRESCRIPTIONS.csv`, and `PROCEDURES_ICD.csv`, and the following files from MIMIC-IV: `DIAGNOSES_ICD.csv`, `PRESCRIPTIONS.csv`, and `PROCEDURES_ICD.csv`.

3. Download the [drugbank_drugs_info.csv](https://drive.google.com/file/d/1EzIlVeiIR6LFtrBnhzAth4fJt6H_ljxk/view?usp=sharing) and [drug-DDI.csv]( https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing) files, and put them in the `data/input/` directory.

## Process the data

Run the following command to process the data:

```bash
python process.py
```

If things go well, the processed data will be saved in the `data/output/` directory. You can run the models now!

## Run the models

```bash
cd src
bash run_DifMTT.sh
```

You can modify the hyperparameters in `run_DifMTT.sh` according to your needs.

## Project Structure

```
DifMTT/
├── data/
│   ├── input/              # Raw MIMIC data
│   │   ├── DIAGNOSES_ICD.csv
│   │   ├── PRESCRIPTIONS.csv
│   │   ├── PROCEDURES_ICD.csv
│   │   ├── drugbank_drugs_info.csv
│   │   └── drug-DDI.csv
│   ├── output/             # Processed data
│   │   ├── mimic-iii/
│   │   └── mimic-iv/
│   └── process.py          # Data processing script
├── src/
│   ├── models/
│   │   ├── DifMTT.py       # Main model implementation
│   │   └── gnn/            # GNN modules for molecular encoding
│   ├── utils/
│   │   ├── util.py         # Utility functions
│   │   ├── data_loader.py  # Data loading utilities
│   │   └── beam.py         # Beam search utilities
│   ├── baseline/           # Baseline implementations
│   ├── main_DifMTT.py      # Training and evaluation script
│   └── run_DifMTT.sh       # Quick start script
├── log/                    # Training logs and checkpoints
├── figs/                   # Figures for documentation
├── install.sh              # Installation script
├── LICENSE
└── README.md
```



## Citation

If you find this work useful, please cite our paper:
## Acknowledgement

This work is built upon several excellent medication recommendation research. We sincerely thank the authors for their contributions:

- [MoleRec](https://github.com/yangnianzu0515/MoleRec) - Molecular Substructure-aware Medication Recommendation (WWW 2023)
- [SafeDrug](https://github.com/ycq091044/SafeDrug) - Dual Molecular Graph Encoders for Safe Drug Recommendations (IJCAI 2021)
- [GAMENet](https://github.com/sjy1203/GAMENet) - Graph Augmented Memory Networks for Sequential Recommendation (AAAI 2019)
- [COGNet](https://github.com/BarryRun/COGNet) - Copy Or Generate for Drug Recommendation (SIGIR 2022)
- [MICRON](https://github.com/ycq091044/MICRON) - Change Matters for Medication Recommendation (IJCAI 2021)
- [RETAIN](https://github.com/mp2893/retain) - Reverse Time Attention Model for EHR (NeurIPS 2016)
- [LEAP](https://github.com/neozhangthe1/AutoML-Healthcare) - Learn to Prescribe (KDD 2017)
- [RAREMed](https://github.com/zzhUSTC2016/RAREMed) - Reinforced Medication Recommendation with Rare Disease (SIGIR 2024)
- [VITA](https://github.com/jhheo0123/VITA) - Visit-level Interaction Transformer for Medication Recommendation (AAAI 2024)
- [CausalMed](https://github.com/lixiang-222/CausalMed) - Causality-based Medication Recommendation (CIKM 2024)
- [DrugRec/DEPOT](https://github.com/xmed-lab/DrugRec) - Drug Recommendation with Transformers

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions, please feel free to open an issue or contact us.
