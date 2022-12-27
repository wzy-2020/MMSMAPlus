import scipy.io as scio
import pandas as pd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from goatools.obo_parser import GODag
from goatools.anno.idtogos_reader import IdToGosReader
from goatools.semantic import TermCounts, get_info_content,lin_sim
from collections import deque, Counter
import warnings
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
import math

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}

EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',
    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])

# CAFA4 Targets
CAFA_TARGETS = set([
    '287', '3702', '4577', '6239', '7227', '7955', '9606', '9823', '10090',
    '10116', '44689', '83333', '99287', '226900', '243273', '284812', '559292'])


def is_cafa_target(org):
    return org in CAFA_TARGETS


def is_exp_code(code):
    return code in EXP_CODES


def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total = 0
    ru = 0.0
    mi = 0.0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = real_annots[i].intersection(pred_annots[i])
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s

class Ontology(object):

    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)

        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
                if min_n == 0:
                    min_n = n
            self.ic[go_id] = math.log(min_n / n, 2)

    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont

    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while (len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']

    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set


def read_fasta(filename):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    info.append(inf)
                    seq = ''
                inf = line[1:]
            else:
                seq += line
        seqs.append(seq)
        info.append(inf)
    return info, seqs

def MinMax(data):
    (height,width) = data.shape
    normalized_vector = np.zeros((height, width))
    for i in range(width):
        # 按列取最大值最小值
        maxValue, minValue = max(data[:, i]), min(data[:, i])
        try:
            normalized_vector[:, i] = (data[:, i] - minValue) / (maxValue - minValue)

        except ZeroDivisionError as e:
            return 0, e

    return normalized_vector

class DataGenerator(object):

    def __init__(self, batch_size, is_sparse=False):
        self.batch_size = batch_size
        self.is_sparse = is_sparse

    def fit(self, inputs, targets=None):
        self.start = 0
        self.inputs = inputs
        self.targets = targets
        if isinstance(self.inputs, tuple) or isinstance(self.inputs, list):
            self.size = self.inputs[0].shape[0]
        else:
            self.size = self.inputs.shape[0]
        self.has_targets = targets is not None

    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            if isinstance(self.inputs, tuple) or isinstance(self.inputs, list):
                res_inputs = []
                for inp in self.inputs:
                    if self.is_sparse:
                        res_inputs.append(
                            inp[batch_index, :].toarray())
                    else:
                        res_inputs.append(inp[batch_index, :])
            else:
                if self.is_sparse:
                    res_inputs = self.inputs[batch_index, :].toarray()
                else:
                    res_inputs = self.inputs[batch_index, :]
            self.start += self.batch_size
            if self.has_targets:
                if self.is_sparse:
                    labels = self.targets[batch_index, :].toarray()
                else:
                    labels = self.targets[batch_index, :]
                return (res_inputs, labels)
            return res_inputs
        else:
            self.reset()
            return self.next()


def pass_config(device,args,opt):
    opt.device = device
    opt.namespace = args.namespace
    opt.net_type = args.net_type
    opt.feats_type = args.feats_type
    opt.num_epochs = args.num_epochs

def read_pkl(pkl_path):
    f = open(pkl_path, 'rb')
    df = pickle.load(f)
    # 全显示
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_colwidth', None)
    return df


def read_list(list_path):
    seq_list = pd.read_csv(list_path,names=["index"])
    seq_list = seq_list['index'].values.tolist()
    return seq_list

def save_list(a_list,a_path):
    with open(a_path, 'w') as fin:
        fin.write('\n'.join(a_list))
# 查看 mat矩阵文件
def read_mat(mat_file):
    filepro =  scio.loadmat(mat_file)
    transmat = filepro['A']
    transmat = np.array(transmat)
    return transmat

# def read_fasta(fasta_path, split_char, id_field):
#     '''
#         Reads in fasta file containing multiple sequences.
#         Returns dictionary of holding multiple sequences or only single
#         sequence, depending on input file.
#     '''
#
#     sequences = dict()
#     with open(fasta_path, 'r') as fasta_f:
#         for line in fasta_f:
#             # get uniprot ID from header and create new entry
#             if line.startswith('>'):
#                 uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
#                 sequences[uniprot_id] = ''
#             else:
#                 # repl. all whie-space chars and join seqs spanning multiple lines
#                 sequences[uniprot_id] += ''.join(line.split()).upper()
#     return sequences

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def tv_split(data_file, split):
    df = pd.read_pickle(data_file)
    # Split train/valid
    n = len(df)
    index = np.arange(n)
    train_n = int(n * split)
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = df.iloc[index[:train_n]]
    valid_df = df.iloc[index[train_n:]]

    return train_df, valid_df

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_loss(n,save_dir):
    y = []
    enc = np.load(save_dir+'epoch_{}.npy'.format(n))
    tempy = list(enc)
    y += tempy

    x = range(0, len(y))
    plt.plot(x, y, '.-')
    plt_title = 'BATCH_SIZE = 64; LEARNING_RATE:0.00005'
    plt.title(plt_title)
    plt.xlabel('epoch')
    plt.ylabel('LOSS')
    plt.savefig(save_dir+"epoch_{}".format(n))
    # plt.show()

# slide window
def get_data(d, wind_size):
    print('Building data ...')
    mats = []
    # pssm + bert
    pader = [0.0] * 84
    pssm = d["pssm"]
    bert = d["bert"]
    print(bert.shape)
    print(pssm.shape)
    length = len(d["sequence"])
    if len(d["sequence"]) > 2000:
        length = 2000
    mat = []
    for i in range(length):
        tmp = []
        for r in range(i - int(wind_size / 2), i + int(wind_size / 2) + 1):
            # 当 win = 7 ， 时 前3行都为0
            if r < 0:
                tmp.extend(pader)
            elif r >= length:
                tmp.extend(pader)
            else:
                tmp.extend(pssm[r])
                tmp.extend(bert[r])
        mat.append(tmp)

    mat = np.array(mat)
    return mat

# def load_all_files(opt):
#     names_dir = opt.namespace_dir
#     terms_file = names_dir + "/{}.txt".format(opt.namespace)
#     terms = read_list(terms_file)
#     icvec_file = names_dir + "/icVec.npy"
#     icvec = np.load(icvec_file).astype(np.float32)
#     GOXfile = names_dir + "/goxfile.npy"
#     transMatFile = names_dir + "/RWLimSim.mat"
#     train_names = read_list(names_dir + "/train.names")
#     train_names = pd.DataFrame(train_names, columns=["index"])
#     train_names, valid_names = tv_splist(train_names, opt.train_size)
#     train_names = train_names['index'].values.tolist()
#     valid_names = valid_names['index'].values.tolist()
#     test_names = read_list(names_dir + "/test.names")
#
#     return terms,train_names,valid_names,test_names,icvec,GOXfile,transMatFile








