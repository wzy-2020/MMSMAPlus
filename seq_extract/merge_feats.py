import sys,os
sys.path.append('Utility')

from utils_encoder import *


dict_list = ['onehot', 'pssm', 'blosum62', 'zscalue', 'opf', 'char2vec', 'bert']
def merge_feats(feats_dir,save_dir):
    protein_list = os.listdir(feats_dir)
    count = 0
    for pdir in protein_list:
        if count % 100 == 0:
            print(f"count:{count}")
        count += 1
        protein = pdir.split('.')[0]
        new_feats = dict()
        d = pickle.load(open(feats_dir + '/' + pdir, "rb"))
        new_feats['sequence'] = d['sequence']
        new_feats['pssm'] = d['pssm']
        new_feats['bert'] = d['bert']

        new_feats['Y'] = d['Y']

        with open(save_dir + f'/{protein}.pkl', "wb") as f:
            pickle.dump(new_feats, f)


def check_feat(feats_dir,save_dir):
    protein_list = os.listdir(feats_dir)
    for id,pdir in enumerate(protein_list):
        if id % 500 == 0:
            print(f"pos:{id}")
        protein = pdir.split('.')[0]
        d = pickle.load(open(feats_dir + f'/{protein}.pkl', "rb"))
        new_feats = d
        opf = to_OPF_10bit(d['sequence'])
        new_feats['opf'] = opf

        with open(save_dir + f'/{protein}.pkl', "wb") as f:
            pickle.dump(new_feats, f)



def main():
    feats_dir = 'E:/Datasets/cafa_feats'
    save_dir = 'F:/CAFA3/feats'
    train_df = pd.read_pickle('data-cafa3/train_data.pkl')
    test_df = pd.read_pickle('data-cafa3/test_data.pkl')
    data_df = pd.concat([train_df,test_df],ignore_index=True)
    ont_list = ['MFO','BPO','CCO']
    merge_feats(feats_dir,save_dir)

    return


if __name__ == '__main__':
    main()





