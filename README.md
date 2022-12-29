# MMSMA and MMSMAPlus

This repository contains script which were used to build and train the MMSMAPlus model together with the scripts for evaluating the model's performance.

# Get Started

## Requirements

Please, install the following packages:

- numpy
- torch-1.8.0
- torchversion-0.9.0
- transformers-4.5.1
- tqdm

Secondly, install [diamond](https://github.com/bbuchfink/diamond) program on your system (diamond command should be available)

## Download pretrain models

seq_extract/LM/bert ([GoogleDrive](https://drive.google.com/file/d/1HoUXtxqmz0SYDVXrA3ETmANH7UPGg5DI/view?usp=share_link))

## Traning

To train CAFA3 with this repo:

```python
python main.py
```

The superparameters are set in the '**pser_args**' function of main.py, too.

