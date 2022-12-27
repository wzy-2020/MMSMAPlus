import pandas as pd
import numpy as np

input_dim_dict = {'onehot': 21, 'pssm': 20, 'bert': 292, 'zscalue': 5, 'opf': 10}


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

def read_list(list_path):
    seq_list = pd.read_csv(list_path,names=["index"])
    seq_list = seq_list['index'].values.tolist()
    return seq_list


def save_list(a_list,a_path):
    with open(a_path, 'w') as fin:
        fin.write('\n'.join(a_list))
# 查看 mat矩阵文件
# def read_mat(mat_file):
#     filepro = scio.loadmat(mat_file)
#     transmat = filepro['A']
#     transmat = np.array(transmat)
#     return transmat
def get_data(path,ont,split=0.95):

    namespace_dir = path + "/" +ont
    terms = np.loadtxt(namespace_dir + f"/{ont}.txt", dtype=str)
    num_classes = len(terms)
    terms_dict = {v: i for i, v in enumerate(terms)}
    terms_df = pd.DataFrame({'terms': terms})
    terms = terms_df['terms'].values.flatten()

    train_df = pd.read_pickle(namespace_dir + '/train_data.pkl')
    test_df = pd.read_pickle(namespace_dir + '/test_data.pkl')
    n = len(train_df)
    index = np.arange(n)
    np.random.shuffle(index)
    train_size = int(n * split)
    valid_df = train_df.iloc[index[train_size:]]
    train_df = train_df.iloc[index[:train_size]]

    print(f"[*] train data: {len(train_df)} \t valid data : {len(valid_df)} \t  test data: {len(test_df)} \t    num_classes : {num_classes}")

    return terms_dict,terms,train_df,valid_df,test_df,num_classes

# slide window
def get_window_data(d, wind_size):
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

def pass_config(device,args,opt):
    opt.device = device
    opt.namespace = args.namespace
    opt.net_type = args.net_type
    opt.feats_type = args.feats_type
    opt.num_epochs = args.num_epochs
    input_dim, input_dims =None, []
    if '_' in args.feats_type:
        feats_list = args.feats_type.split('_')
        for char in feats_list:
            if char == 'B':
                input_dim = 292
            elif char == 'O':
                input_dim = 21
            elif char == 'P':
                input_dim = 20
            elif char == 'Z':
                input_dim = 5
            input_dims.append(input_dim)
    else:
        input_dim = input_dim_dict[args.feats_type]
    return input_dim,input_dims

