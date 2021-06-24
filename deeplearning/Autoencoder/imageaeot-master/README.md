# ImageAEOT

This repository contains code adapted from the paper, "Predicting Cell Lineages using Autoencoders and Optimal
Transport." 

## Setup and requirements
Dependencies are listed in environment.yml file and can be installed using Anaconda/Miniconda:
```
conda env create -f environment.yml
```
Autoencoder models were trained on an NVIDIA Tesla A100 GPU.

## Usage

To train the autoencoder on image files:
```
python run_train.py --datadir <path/to/image/directory> --save-dir <path/to/save/directory>
```

To extract AE features:
```
python get_features.py --datadir <path/to/image/directory> --save-dir <path/to/save/directory> --pretrained-file <path/to/checkpoint.pth> --ae-features
```