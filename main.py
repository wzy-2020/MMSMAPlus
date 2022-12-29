import argparse
from utils import *
from collections import Counter
import os
import random
import torch
from baselines import CNN1D_DeepGoPlus
from networks import MSMA
from datasets1 import CNN1DDataset,cnn1d_collate
from model import train,test,load_checkpoint
from data_utils import get_data,pass_config
from torch.utils.data import DataLoader as pyDataLoader
from config import opt

def fix_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--phase', dest='phase', default='train')  # 'train' / 'test'
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=12)  # cafa : 15 / homo : 30
    parser.add_argument('--namespace', dest='namespace', default="mf")
    parser.add_argument('--datasets', dest='datasets', default="cafa3")      # cafa / homo
    parser.add_argument('--net_type', dest='net_type', default="MSMA") #  DMPF / MLP /DeepGOCNN / DeepGOA / MV_Models_1
    parser.add_argument('--feats_type', dest='feats_type', default="onehot") # bert onehot pssm word2vec / B_O_P_W / B_P_O
    parser.add_argument('--namespace_dir', dest='namespace_dir', default="data/")
    parser.add_argument('--feats_dir', dest='feats_dir', default='E:/Datasets/cafa_feats')  # cafa_feats / homo_feats
    parser.add_argument('--model_dir', dest='model_dir')
    parser.add_argument('--out_file', dest='out_file')
    return parser.parse_args()


def main(**kwargs):
    num_head = 4
    opt.parse(kwargs)
    args = pser_args()
    go = Ontology(f'data/{args.datasets}/go.obo', with_rels=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataPre
    input_dim, input_dims = pass_config(device, args, opt)
    terms_dict, terms, train_df, valid_df, test_df, num_classes = get_data('data/' + args.datasets, args.namespace,
                                                                           split=0.9)

    args.model_dir = "models/" +args.datasets+"/"+ args.namespace.upper() + "/" + args.net_type + '/' + args.feats_type
    args.out_file = args.model_dir + "/prediction.pkl"
    print("[*] Model will saving to :", args.model_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    logger_file = "models/" +args.datasets+"/"+ args.namespace.upper() + "/{}_{}_{}_{}_result.txt".format(args.datasets,args.namespace.upper(),args.net_type,args.feats_type)
    F_txt = open(logger_file, "a+")

    # Init network
    if args.net_type == "DeepGOCNN":
        net = CNN1D_DeepGoPlus(input_dim=input_dim, num_filters=16 * [512],num_classes=num_classes).to(device)

    elif args.net_type == "MSMA":
        net = MSMA(input_dim=input_dim,hidden_dim=512, num_classes=num_classes,num_head=num_head).to(device)


    print(f"{now()} {args.net_type} init model finished")
    print("[*] Number of model parameters: ",sum(p.numel() for p in net.parameters() if p.requires_grad))
    print("[*] Number of model parameters: ",sum(p.numel() for p in net.parameters() if p.requires_grad),file=F_txt)

    print(net)
    
    train_set = CNN1DDataset(df=train_df, feats_dir=args.feats_dir, terms_dict=terms_dict, feats_type=args.feats_type)
    train_loader = pyDataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                collate_fn=cnn1d_collate)

    valid_set = CNN1DDataset(df=valid_df, feats_dir=args.feats_dir, terms_dict=terms_dict, feats_type=args.feats_type)
    valid_loader = pyDataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                                collate_fn=cnn1d_collate)

    test_set = CNN1DDataset(df=test_df, feats_dir=args.feats_dir, terms_dict=terms_dict, feats_type=args.feats_type)
    test_loader = pyDataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                               collate_fn=cnn1d_collate)

    criterion = torch.nn.BCELoss().to(device)

    print(f"[*] train data: {len(train_df)}\t valid data: {len(valid_df)}\t test data: {len(test_df)}\t num_classes: {num_classes}")
    ckpt_dir = args.model_dir + '/checkpoint'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if args.phase == "train":
        print(f'=============== Training in the train set \t{now()}===============')
        print(f'=============== Training in the train set \t{now()}===============', file=F_txt)
        # Training and validation
        train(opt=opt, net=net, criterion=criterion, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,ckpt_dir=ckpt_dir,
                            F_txt=F_txt)

    # Testing
    model_fmax = []
    model_list = ['test']
    train_df = pd.concat([train_df, valid_df], ignore_index=True)
    print(f"Training Size:{len(train_df)}\t Testing Size:{len(test_df)}")
    num_epoch = 0
    for name in model_list:
        args.model_dir = ckpt_dir + f'/model_{name}.pth.tar'
        print(args.model_dir)
        epoch_num = load_checkpoint(net, filename=args.model_dir)
        if epoch_num != num_epoch:
            num_epoch = epoch_num
        else:
            break
        epoch_num,test_loss, test_rocauc,test_fmax, y_true, y_pred_sigm = test(device=device, net=net, criterion=criterion, model_file=args.model_dir, test_loader=test_loader)
        print(f"[*] Loaded checkpoint at epoch {epoch_num} for {name}ing:",file=F_txt)
        print("Test ROC AUC:{:.4f}\tTest Fmax:{:.4f}".format(test_rocauc,test_fmax), file=F_txt)
        # save prediction.pkl
        test_df['labels'] = list(y_true)
        test_df['preds'] = list(y_pred_sigm)
        test_df.to_pickle(args.out_file)

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
        go_set = go.get_namespace_terms(NAMESPACES[args.namespace])
        go_set.remove(FUNC_DICT[args.namespace])

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
                for j, score in enumerate(y_pred_sigm[i]):
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
        model_fmax.append(fmax)
        print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}')
        print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}',file=F_txt)

        F_txt.flush()


    F_txt.close()


if __name__ == '__main__':
    fix_random_seed(2021)
    main()





