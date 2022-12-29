import os
import pickle
import pandas as pd
import torch
from torch import nn
from torch import optim
from transformers import AutoTokenizer,AutoModel
from transformers import WEIGHTS_NAME,CONFIG_NAME
from torch.utils.data import DataLoader as pyDataLoader
from config import opt
import numpy as np
import math


def split_sequences(sequence,length=1024):
    sub_seq_list = []
    size = len(sequence)
    sub_num = int(size / length)  # sub_num = 3
    for i in range(sub_num):
        sub_seq = sequence[i * length:(i + 1) * length]
        sub_seq_list.append(sub_seq)
    if size > sub_num * length:
        tmp = sequence[sub_num * length:]
        sub_seq_list.append(tmp)

    return sub_seq_list


def main(df_file='cafa3/test_df.pkl',save_dir='G:\Datasets\\cafa3\\test'):
    verbose = True
    max_chars = 15000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_weights = 'LM/bert'
    model = AutoModel.from_pretrained(pretrained_weights).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)
    seq_anno_df = pd.read_pickle(df_file)
    print(seq_anno_df.columns)
    seq_dict = dict()

    for index, row in enumerate(seq_anno_df.itertuples()):
        seq = row.sequences
        protein = row.proteins
        seq_dict[protein] = seq

    ####################### Sort sequences ###############################
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(seq_dict[kv[0]]))
    print(len(seq_dict))

    if verbose: print('Total number of sequences: {}'.format(len(seq_dict)))

    batch = list()
    fail_batch = list()
    length_counter = 0
    success = list()
    for index, (identifier, sequence) in enumerate(seq_dict):
        print('index:', index)
        batch.append((identifier, sequence))
        length_counter += len(sequence)

        if length_counter > max_chars or len(sequence) > max_chars / 2 or index == len(seq_dict) - 1:
            # 将序列切割为字符列表
            sentence_lists = [' '.join(seq) for _,seq in batch]
            seq_lists = [seq for _,seq in batch]
            runtime_error = False
            # 对当前batch 中的序列进行 bert 编码
            #######################  Batch-Processing #######################
            for batch_idx, (sample_id, seq) in enumerate(batch):  # for each seq in the batch
                # 对当前序列token进行编码
                runtime_error = False
                raw_seq = seq_lists[batch_idx]
                seq_tokens = sentence_lists[batch_idx]
                print(len(raw_seq))
                if len(seq_tokens) > 512:
                    sub_seq_list = split_sequences(seq_tokens,512)
                    sub_embedding_list = []
                    for sub_seq_token in sub_seq_list:
                        ids = tokenizer(sub_seq_token, padding=True, return_tensors="pt")
                        inputs = ids["input_ids"]
                        if torch.cuda.is_available():
                            inputs = inputs.to(device)
                        outputs = model(inputs)
                        embedding = outputs[0].view(-1, 1024)  # (1,L,1024) -> (L,1024)
                        embedding = embedding.cpu().detach().numpy().squeeze()
                        sub_embedding_list.append(embedding[1:embedding.shape[0]-1])
                    embeddings = np.vstack(sub_embedding_list)
                else:
                    ids = tokenizer(seq_tokens, padding=True, return_tensors="pt")
                    inputs = ids["input_ids"]
                    if torch.cuda.is_available():
                        inputs = inputs.to(device)
                    outputs = model(inputs)
                    embedding = outputs[0].view(-1, 1024)  # (1,L,1024) -> (L,1024)
                    embedding = embedding.cpu().detach().numpy().squeeze()
                    embeddings = embedding[1:embedding.shape[0]-1]

                # 若未运行超时 ，就将当前 embedding 保存下来
                if runtime_error == False:
                    try:
                        if verbose: print('Writing embeddings to: {}'.format(sample_id))

                        with open(save_dir + '\\{}.npz'.format(sample_id), 'wb') as f:
                            pickle.dump(embeddings, f)
                    except ZeroDivisionError:
                        print('Error: Embedding dictionary is empty!')

                if runtime_error:

                    print('Single sequence processing not possible. Skipping seq. ..' +
                          'Consider splitting the sequence into smaller seqs or process on CPU.')

            ################## Reset batch ####################
            batch = list()
            length_counter = 0
            if verbose: print('.', flush=True, end='')

    if verbose: print('\nTotal number of embeddings: {}'.format(len(success)))
    if verbose: print('\nTotal number of Fail embeddings: {}'.format(len(fail_batch)))

    return None

if __name__ == '__main__':
    main()



