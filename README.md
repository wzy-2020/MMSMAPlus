# MMSMAPlus

This repository contains script which were used to build and train the MMSMAPlus model together with the scripts for evaluating the model's performance.

# Get Started

## Dependency

Please, install the following packages:

- numpy
- torch-1.8.0
- torchversion-0.9.0
- transformers-4.5.1
- tqdm

Secondly, install [diamond](https://github.com/bbuchfink/diamond) program on your system (diamond command should be available)

## Content

- ./data: the dataset with label and sequence, including CAFA3 and Homo datasets.
- ./seq_extract: extraction of protein sequences with deep semantic view features.

## Usage

### step 1:

Firstly，please download pretrain model ([Fine-tuning ProtBERT](https://drive.google.com/file/d/1HoUXtxqmz0SYDVXrA3ETmANH7UPGg5DI/view?usp=share_link)), and move it to *seq_extract/LM/bert*. Then execute the "generate_bert.py" script,it will save 'P32558.pkl',this includes the deep semantic features of protein sequences. 

### step 2:

The "main.py" script is called to train different view versions of MSMA as a classifier to generate preliminary results for the future test data. These results will be saved as "prediction.pkl". 

The hyperparameters are set in the '**pser_args**' function of main.py, too.

### step 3:

Based on the output from step 2, call the "AWF_net.py" script to train a multi-view adaptive decision model.

### step 4:

Integrate homology-based  method prediction by running "evaluate_integrated.py".

## Contact

If you have any suggestions or questions, please email me at 6201613051@stu.jiangnan.edu.cn.



