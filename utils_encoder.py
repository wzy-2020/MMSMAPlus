import pickle
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing

MAXLEN = 2000
AALETTER = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AANUM = len(AALETTER)
AAINDEX = dict()
for i in range(len(AALETTER)):
    AAINDEX[AALETTER[i]] = i + 1
INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X', '*'])

min_max_scaler = preprocessing.MinMaxScaler()

def to_onehot(seq,start=0):
    # binary
    l = min(MAXLEN, len(seq))
    onehot = np.zeros((l, 21), dtype=np.int32)
    for i in range(start, start + l):
        onehot[i, AAINDEX.get(seq[i - start], 0)] = 1
    onehot[0:start, 0] = 1
    onehot[start + l:, 0] = 1
    return onehot

def to_zscale(seq,strat=0):
    l = min(MAXLEN,len(seq))
    zscale = {
        'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
        'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
        'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
        'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
        'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
        'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
        'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
        'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
        'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
        'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
        'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
        'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
        'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
        'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
        'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
        'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
        'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
        'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
        'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
        'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
        '-': [0.00, 0.00, 0.00, 0.00, 0.00],  # -
    }
    embedding = np.zeros((l,5),dtype=np.float32)
    for id,aa in enumerate(seq[strat:strat+l]):
        if aa in zscale:
            embedding[id,:] = zscale[aa]
        else:
            embedding[id,:] = zscale['-']

    embedding = min_max_scaler.fit_transform(embedding)
    return embedding

def to_AAIndex(seq,start=0):
    l = min(len(seq),MAXLEN)
    AA = 'ARNDCQEGHILKMFPSTWYV'
    fileAAindex = 'data/AAindex.txt'
    with open(fileAAindex) as f:
        records = f.readlines()[1:]

    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    # or other properties
    props = AAindexName
    if props:
        tmpIndexNames = []
        tmpIndex = []
        for p in props:
            if AAindexName.index(p) != -1:
                tmpIndexNames.append(p)
                tmpIndex.append(AAindex[AAindexName.index(p)])
        if len(tmpIndexNames) != 0:
            AAindexName = tmpIndexNames
            AAindex = tmpIndex

    code = []
    embedding = np.zeros((l, 531),dtype=np.float32)
    for id_row, aa in enumerate(seq[start:start+l]):
        if aa not in AA:
            for j in AAindex:
                code.append(0)
            continue
        for id_col, j in enumerate(AAindex):
            embedding[id_row, id_col] = j[index[aa]]

    embedding = min_max_scaler.fit_transform(embedding)

    return embedding

def to_Blosum62(seq,strat=0):
    l = min(MAXLEN,len(seq))
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
        '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    }
    embedding = np.zeros((l, 20), dtype=np.float32)
    for id, aa in enumerate(seq[strat:strat+l]):
        if aa in blosum62:
            embedding[id, :] = blosum62[aa]
        else:
            embedding[id, :] = blosum62['-']

    feature = np.zeros([(l), 20])
    axis_x = 0
    for i in range(len(seq)):
        raw_blosum = embedding[i]
        axis_y = 0
        for j in raw_blosum:
            feature[axis_x][axis_y] = (1 / (1 + math.exp(-float(j))))
            axis_y += 1
        axis_x += 1

    return feature

def to_OPF_7bit_type_1(seq,start=0):
    # binary
    l = min(MAXLEN,len(seq))

    physicochemical_properties_list = [
        'ACFGHILMNPQSTVWY',
        'CFILMVW',
        'ACDGPST',
        'CFILMVWY',
        'ADGST',
        'DGNPS',
        'ACFGILVW',
    ]
    embedding = np.zeros((l, 7), dtype=np.int32)
    for i, aa in enumerate(seq[start:start+l]):
        for j,pp in enumerate(physicochemical_properties_list):
            if aa in pp:
                embedding[i,j] = 1
            else:embedding[i,j] = 0

    return embedding

def to_OPF_7bit_type_2(seq,start=0):
    # binary
    l = min(MAXLEN,len(seq))

    physicochemical_properties_list = [
        'DE',
        'AGHPSTY',
        'EILNQV',
        'AGPST',
        'CEILNPQV',
        'AEHKLMQR',
        'HMPSTY',
    ]
    embedding = np.zeros((l, 7), dtype=np.int32)
    for i, aa in enumerate(seq[start:start+l]):
        for j,pp in enumerate(physicochemical_properties_list):
            if aa in pp:
                embedding[i,j] = 1
            else:embedding[i,j] = 0

    return embedding

def to_OPF_7bit_type_3(seq,start=0):
    # binary
    l = min(MAXLEN,len(seq))

    physicochemical_properties_list = [
        'KR',
        'DEKNQR',
        'FHKMRWY',
        'DEHKNQR',
        'FHKMRWY',
        'CFITVWY',
        'DEKNRQ',
    ]
    embedding = np.zeros((l, 7), dtype=np.int32)
    for i, aa in enumerate(seq[start:start+l]):
        for j,pp in enumerate(physicochemical_properties_list):
            if aa in pp:
                embedding[i,j] = 1
            else:embedding[i,j] = 0

    return embedding

def to_OPF_10bit(seq,start=0):
    # binary
    l = min(MAXLEN,len(seq))

    physicochemical_properties_list = [
        'FYWH',
        'DE',
        'KHR',
        'NQSDECTKRHYW',
        'AGCTIVLKHFYWM',
        'IVL',
        'ASGC',
        'KHRDE',
        'PNDTCAGSV',
        'P',
    ]
    embedding = np.zeros((l, 10), dtype=np.int32)
    for i, aa in enumerate(seq[start:start+l]):
        for j,pp in enumerate(physicochemical_properties_list):
            if aa in pp:
                embedding[i,j] = 1
            else:embedding[i,j] = 0

    return embedding

