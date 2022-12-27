import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils_encoder import *
import joblib

MAXLEN = 2000

class CNN1DDataset(Dataset):
    def __init__(self,df, feats_dir, terms_dict,feats_type='onehot'):
        # Initialize data
        self.df = df
        self.feats_dir = feats_dir
        self.terms_dict = terms_dict
        self.feats_type = feats_type
        self.bert_pca = joblib.load('data/cafa3/pca_bert_n0.95.m')

    def __len__(self):
        # Get total number of samples
        return len(self.df)

    def __getitem__(self, index):
        # Load sample
        batch_df = self.df.iloc[index]
        name = batch_df.proteins
        seq = batch_df.sequences
        prop_annotations = batch_df.annotations
        terms_dict = self.terms_dict
        sub_x1 = None
        sub_x2 = None
        sub_x3 = None
        sub_x4 = None
        sub_x5 = None
        seqlen = len(seq)
        if seqlen > MAXLEN:
            seqlen = MAXLEN

        if self.feats_type == 'onehot':
            onehot = to_onehot(seq)
            features = onehot.astype(np.float32).T

        elif self.feats_type == 'pssm':
            pssm = pickle.load(open(self.feats_dir + '/' + name + '.pkl', 'rb'))['pssm']
            features = pssm.astype(np.float32).T

        elif self.feats_type == 'zscalue':
            zscale = to_zscale(seq)
            features = zscale.astype(np.float32).T

        elif self.feats_type == 'opf':
            opf = to_OPF_10bit(seq)
            features = opf.astype(np.float32).T

        elif self.feats_type == 'bert':
            bert = pickle.load(open(self.feats_dir + '/' + name + '.pkl', 'rb'))['bert']
            features = self.bert_pca.transform(bert).astype(np.float32).T

        # Get labels (N)
        labels = np.zeros(len(terms_dict), dtype=np.int32)
        for g_id in prop_annotations:
            if g_id in terms_dict:
                labels[terms_dict[g_id]] = 1

        return features, seqlen,labels,sub_x1,sub_x2,sub_x3,sub_x4,sub_x5

def cnn1d_collate(batch):

    feats = [item[0] for item in batch]
    lengths = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    feats1 = [item[3] for item in batch]
    feats2 = [item[4] for item in batch]
    feats3 = [item[5] for item in batch]
    feats4 = [item[6] for item in batch]
    feats5 = [item[7] for item in batch]

    # max_len = 2000
    max_len = max(lengths)

    # Pad data to max sequence length in batch
    feats_pad = [np.pad(item, ((0,0),(0,max_len-lengths[i])), 'constant') for i, item in enumerate(feats)]
    x = torch.from_numpy(np.array(feats_pad))

    x1 = None
    x2 = None
    x3 = None
    x4 = None
    x5 = None
    # 如果 feats1 不是 None
    if hasattr(feats1[0], 'dtype'):
        feats1_pad = [np.pad(item, ((0, 0), (0, max_len - lengths[i])), 'constant') for i, item in enumerate(feats1)]
        x1 = torch.from_numpy(np.array(feats1_pad))

    # 如果 feats2 不是 None
    if hasattr(feats2[0], 'dtype'):
        feats2_pad = [np.pad(item, ((0, 0), (0, max_len - lengths[i])), 'constant') for i, item in enumerate(feats2)]
        x2 = torch.from_numpy(np.array(feats2_pad))
    if hasattr(feats3[0], 'dtype'):
        feats3_pad = [np.pad(item, ((0, 0), (0, max_len - lengths[i])), 'constant') for i, item in enumerate(feats3)]
        x3 = torch.from_numpy(np.array(feats3_pad))

    if hasattr(feats4[0], 'dtype'):
        feats4_pad = [np.pad(item, ((0, 0), (0, max_len - lengths[i])), 'constant') for i, item in enumerate(feats4)]
        x4 = torch.from_numpy(np.array(feats4_pad))

    if hasattr(feats5[0], 'dtype'):
        feats5_pad = [np.pad(item, ((0, 0), (0, max_len - lengths[i])), 'constant') for i, item in enumerate(feats5)]
        x5 = torch.from_numpy(np.array(feats5_pad))
    return CustomData(x=x, x1=x1, x2 = x2,x3=x3,x4 = x4,x5 = x5,y=torch.from_numpy(np.array(labels)))

class SeqDataset(Dataset):
    def __init__(self, names, feats_dir, terms_dict,feats_type='onehot'):
        # Initialize data
        self.names = names
        self.feats_dir = feats_dir
        self.terms_dict = terms_dict
        self.feats_type = feats_type

    def __len__(self):
        # Get total number of samples
        return len(self.names)

    def __getitem__(self, index):
        # Load sample
        name = self.names[index]
        # print(name)
        terms_dict = self.terms_dict
        # Load pickle file with dictionary containing embeddings (LxF), sequence (L) and labels (1xN)
        d = pickle.load(open(self.feats_dir + '/' + name + '.pkl', 'rb'))
        seq = d['sequence']
        seqlen = len(seq)
        if seqlen > 2000:
            seq = seq[:2000]
            seqlen = 2000

        # Get labels (N)
        prop_annotations = d['Y']
        labels = np.zeros(len(terms_dict), dtype=np.int32)
        for g_id in prop_annotations:
            if g_id in terms_dict:
                labels[terms_dict[g_id]] = 1

        return seq, seqlen,labels

def seq_collate(batch):

    feats = [item[0] for item in batch]
    lengths = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    return CustomData(x=feats, y=torch.from_numpy(np.array(labels)))

class MLPDataset(Dataset):
    def __init__(self, names_file, feats_dir, terms_dict,feats_type='embeddings'):
        # Initialize data
        self.names = list(np.loadtxt(names_file, dtype=str))
        self.feats_dir = feats_dir
        self.terms_dict = terms_dict
        self.feats_type = feats_type

    def __len__(self):
        # Get total number of samples
        return len(self.names)

    def __getitem__(self, index):
        # Load sample
        name = self.names[index]
        terms_dict = self.terms_dict

        # Get protein-level features
        d = pickle.load(open(self.feats_dir + '\\' + name + '.pkl', 'rb'))
        X = d["X"]

        # Select features type
        if self.feats_type == 'onehot':
            # onehot = d["onehot"].toarray().astype(np.float32).T
            features = d["onehot"].astype(np.float32).T
        elif self.feats_type == 'pssm':
            features = X[:, :20].astype(np.float32).T

        elif self.feats_type == 'embedding':
            features = X[:, 20:].astype(np.float32).T

        elif self.feats_type == 'X':
            features = X[:, :20].astype(np.float32).T
            x1 = X[:, 20:].astype(np.float32).T

        else:
            print('[!] Unknown features type, try "embeddings" or "onehot".')
            exit(0)

        features = np.mean(features, 1)

        # Get labels (N)
        prop_annotations = d['Y']
        labels = np.zeros(len(terms_dict), dtype=np.int32)
        for g_id in prop_annotations:
            if g_id in terms_dict:
                labels[terms_dict[g_id]] = 1

        return features, labels

def mlp_collate(batch):
    # Get data, label and length (from a list of arrays)
    feats = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    return CustomData(x=torch.from_numpy(np.array(feats)), y=torch.from_numpy(np.array(labels)))

class CustomData(Data):
    def __init__(self, x=None, x1 = None,x2 = None,x3=None,x4=None,x5=None,y=None, **kwargs):
        super(CustomData, self).__init__()

        self.x = x
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.y = y

        for key, item in kwargs.items():
            self[key] = item