BLAST = 'D:/Bioinformatics_tools/blast/bin/psiblast'
BLAST_DB = 'D:/Bioinformatics_tools/blast/db/swissprot'

ALPHABET = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
def str2bool(value):
    return value.lower() == 'true'

class Config():
    phase = 'train'
    namespace = 'cc'
    feats_type = "onehot"
    net_type = "cnn"
    namespace_dir = 'data/' + namespace
    feats_dir = "data/feats"
    model_dir = "models"
    bert_path = "LM_encoder/premodels/prot_bert_bfd"

    use_gpu = False
    seed = 1995
    gpu_id = 0

    dropout = 0.5
    num_epochs = 10
    num_view = 1
    view_list = ['onehot','pssm']
    gamma = 3

    train_size = 0.005
    learning_rate = 0.0005
    lr_sched = True
    batch_size = 32
    num_classes = 289
    device = "cuda:0"


# 更新函数
def parse(self, kwargs):
    '''
    根据字典 kwargs 更新 config 函数
    user can update the default hyperparamter
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            raise Exception('opt has No key: {}'.format(k))
        setattr(self, k, v)

    # # 打印配置信息
    # print('*************************************************')
    # print('user config:')
    # for k, v in self.__class__.__dict__.items():
    #     if not k.startswith('__'):
    #         print("{} => {}".format(k, getattr(self, k)))
    #
    # print('*************************************************')


Config.parse = parse
opt = Config()
