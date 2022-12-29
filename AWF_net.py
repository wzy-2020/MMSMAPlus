import os
from tqdm import tqdm
from utils import *
import torch
import torch.optim as optim
import torch.nn as nn
from datasets1 import CNN1DDataset,cnn1d_collate,CustomData
from torch.utils.data import DataLoader as pyDataLoader
from torch.utils.data import Dataset,TensorDataset
from torch.nn import Parameter
from evaluation import fmax as compute_fmax
import warnings
warnings.filterwarnings("ignore")


class MyDatasets(Dataset):

    def __init__(self, n_view,y,x,x1,x2=None,x3=None):
        self.n_view = n_view
        self.y = y
        self.x = x
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def __getitem__(self, index):

        if self.n_view == 2:
            return self.x[index],self.x1[index],self.y[index]
        elif self.n_view == 3:
            return self.x[index], self.x1[index], self.x2[index],self.y[index]
        else:
            return self.x[index], self.x1[index], self.x2[index],self.x3[index],self.y[index]
    def __len__(self):
        return self.x.shape[0]


class SoftWeighted(nn.Module):
    def __init__(self, num_view):
        super(SoftWeighted, self).__init__()
        self.num_view = num_view
        self.weight_var = Parameter(torch.ones(num_view))


    def forward(self, data):
        weight_var = [torch.exp(self.weight_var[i]) / torch.sum(torch.exp(self.weight_var)) for i in range(self.num_view)]
        high_level_preds = 0
        for i in range(self.num_view):
            high_level_preds += weight_var[i] * data[i]

        return high_level_preds,weight_var


# class Perceptron(nn.Module):
#     def __init__(self, num_classes,num_view):
#         super(Perceptron, self).__init__()
#         self.num_view = num_view
#         self.mv_alphas = nn.Sequential(
#             nn.Linear(num_classes * num_view, num_classes),
#             nn.ReLU(),
#             nn.Linear(num_classes,num_view)
#         )
#
#     def forward(self, data):
#
#         multi_view_feats = torch.cat(data, dim=1)
#         fusion_weights = self.mv_alphas(multi_view_feats)
#         fusion_weights = torch.sigmoid(fusion_weights)
#         # fusion_weights = torch.sum(fusion_weights, 0)
#         weight_var = [fusion_weights[i] / torch.sum(fusion_weights) for i in
#                           range(self.num_view)]
#         # print(weight_var)
#         high_level_preds = 0
#         for i in range(self.num_view):
#             high_level_preds += weight_var[i] * x[i]
#
#         return high_level_preds,weight_var

input_dim_dict = {'F':10,'O':21,'P':20,'B':292}
namespace = "mf"
root_dir = "data/cafa3"
feats_type = "B_P_O_F"
# n_view = int(Counter(feats_type)['_']) + 1
n_view = 2
feats_dir = "E:/Datasets/cafa_feats"
BATCH_SIZE = 64
EPOCHS = 32
F_txt = open(f"models/cafa3/{namespace}/{namespace.upper()}_weighted_{feats_type}_result.txt", "a+")

def main(device):
    go = Ontology(f'data/cafa3/go.obo', with_rels=True)
    namespace_dir = root_dir + f'/{namespace}'
    terms_file = namespace_dir + f"/{namespace}.txt"
    terms = np.loadtxt(terms_file, dtype=str)
    num_classes = len(terms)
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(namespace_dir + '/train_data.pkl')
    test_df = pd.read_pickle(namespace_dir + '/test_data.pkl')

    test_labels = np.zeros((len(test_df), num_classes), dtype=np.float32)
    for i, row in enumerate(test_df.itertuples()):
        for go_id in row.annotations:
            if go_id in terms_dict:
                test_labels[i, terms_dict[go_id]] = 1

    # load multiview model

    feats_list = feats_type.split('_')
    input_dims = []
    for char in feats_list:
        input_dims.append(input_dim_dict.get(char))
    Output_list = []
    train_list = []
    name = None
    for feat in feats_list:
        if feat == "B": name = "bert"
        if feat == "P": name = "pssm"
        if feat == "O": name = "onehot"
        if feat == "F": name = "opf"
        # checkpoint = torch.load(f'models/cafa3/{namespace.upper()}/MVPF_MSRA/{name}/checkpoint/model_test.pth.tar')
        # net = MVPF_MSRA(input_dim=input_dim_dict[feat], hidden_dim=512, num_classes=num_classes).to(device)
        # net.load_state_dict(checkpoint['state_dict'])
        preds_file = f'models/cafa3/{namespace}/MVPF_MSRA/{name}/prediction.pkl'
        test_df = pd.read_pickle(preds_file)
        prediction = np.array(test_df['preds'].tolist())
        train_list.append(torch.from_numpy(prediction))
        Output_list.append(prediction)
        # net_list.append(net)


    deal_dataset = MyDatasets(n_view,torch.from_numpy(test_labels).float(),train_list[0],train_list[1],train_list[2],train_list[3])
    test_loader = pyDataLoader(deal_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # net = Perceptron(num_classes,n_view).to(device)
    net = SoftWeighted(n_view).to(device)

    criterion = nn.BCELoss().to(device)

    optimizer = optim.Adam(net.parameters(), lr=5e-3)
    lr_sched = True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_alpha = None
    best_epoch = 0
    f_max = -1
    for epoch in range(EPOCHS):
        test_loss = 0.0
        net.train()
        print(f"[*] Epoch:{epoch + 1} \t Namespace:{namespace.upper()}")
        for data in test_loader:
            data = [x.to(device) for x in data]
            labels = data.pop()
            output,weight_var = net(data)
            loss = criterion(output, labels)
            test_loss += loss.item()
            loss.backward()
            optimizer.step()

        weight_var = None
        with torch.no_grad():  # set all 'requires_grad' to False
            for data in test_loader:
                data = [x.to(device) for x in data]
                labels = data.pop()
                output, weight_var = net(data)
                break

        test_loss /= len(test_loader)
        weight_var = [weight_var[i].item() for i in range(n_view)]
        fusion_pre = np.zeros(Output_list[0].shape, dtype=np.float32)
        for i in range(n_view):
            fusion_pre += weight_var[i] * Output_list[i]

        avg_fmax = compute_fmax(test_labels,fusion_pre , nrThresholds=10)
        avg_loss = criterion(torch.from_numpy(fusion_pre),torch.from_numpy(test_labels))

        if avg_fmax > f_max:
            best_epoch = epoch + 1
            best_alpha = weight_var
            f_max = avg_fmax

        print(f"--- weight val:  {weight_var}")
        print("--- valid Loss:            %.4f" % avg_loss)
        print("--- valid F-score:         %.4f" % avg_fmax)
        if lr_sched:
            scheduler.step(test_loss)

    print(f"[*] Loaded checkpoint at best epoch {best_epoch}:")
    print(f"[*] Loaded checkpoint at best epoch {best_epoch}:", file=F_txt)

    fusion_pre = np.zeros(Output_list[0].shape, dtype=np.float32)
    for i in range(n_view):
        fusion_pre += best_alpha[i] * Output_list[i]

    # evaluate performance
    annotations = train_df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = []
    for i, row in enumerate(test_df.itertuples()):
        annots = set()
        for go_id in row.annotations:
            if go.has_term(go_id):
                annots |= go.get_anchestors(go_id)
        test_annotations.append(annots)

    go.calculate_ic(annotations + test_annotations)
    go_set = go.get_namespace_terms(NAMESPACES[namespace])
    go_set.remove(FUNC_DICT[namespace])

    labels = test_annotations
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))

    fmax = 0.0
    tmax = 0.0
    smin = 1000.0
    precisions = []
    recalls = []
    for t in range(101):
        threshold = t / 100.0
        preds = []
        for i, row in enumerate(test_df.itertuples()):
            annots = set()
            for j, score in enumerate(fusion_pre[i]):
                if score >= threshold:
                    annots.add(terms[j])

            new_annots = set()
            for go_id in annots:
                new_annots |= go.get_anchestors(go_id)
            preds.append(new_annots)

        # Filter classes
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))

        fscore, prec, rec, s = evaluate_annotations(go, labels, preds)
        precisions.append(prec)
        recalls.append(rec)
        print(f'Fscore: {fscore}, S: {s}, threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold

        if smin > s:
            smin = s

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f"n_view:{n_view}\t tweight_var:{weight_var}")
    print(f"n_view:{n_view}\t weight_var:{weight_var}",file=F_txt)
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}')
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}',file=F_txt)
    F_txt.close()



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device)


