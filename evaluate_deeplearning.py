import os
import numpy as np
import pandas as pd
import click as ck
import logging
import pickle
import math
from utils import FUNC_DICT, Ontology, NAMESPACES
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

@ck.command()
@ck.option(
    '--root-data-dir', '-rtrdf', default='data/',
    help='Data file with training features')
@ck.option(
    '--datasets-name', '-dname', default='cafa3',
    help='Test data file')
@ck.option(
    '--ont', '-o', default='cc',
    help='GO subontology (bp, mf, cc)')
def main(root_data_dir, datasets_name, ont):

    net_type = 'SFENet_FPN'
    feats_type = 'pssm'
    train_data_file = root_data_dir + datasets_name + f'/{ont}/train_df.pkl'
    test_data_file = root_data_dir + datasets_name + f'/{ont}/test_df.pkl'
    terms_file = root_data_dir + datasets_name + f'/{ont}/{ont}.txt'
    F_txt = open(f'models/{datasets_name}/{ont}/{ont}_{net_type}results.txt','a+')

    go = Ontology(root_data_dir + datasets_name+'/go.obo', with_rels=True)
    # single
    predictions = pickle.load(open(f'models/{datasets_name}/{ont}/{net_type}/{feats_type}/prediction.pkl', 'rb'))['y_pred']
    # multi-view
    feats_list = ['zscalue', 'onehot', 'pssm', 'bert']
    num_feats = len(feats_list)
    Output_list = []
    for feat in feats_list:
        prediction = pickle.load(open(f'models/{datasets_name}/{ont}/{net_type}/{feat}/prediction.pkl', 'rb'))['y_pred']
        Output_list.append(prediction)

    weight_var = [ 0.16918921 , 0.520175 ,  -0.16471961 ,-0.08892302]
    fusion_pre = np.zeros(Output_list[0].shape,dtype=np.float32)
    for i in range(num_feats):
        fusion_pre += weight_var[i] * Output_list[i]

    predictions = fusion_pre
    terms = np.loadtxt(terms_file, dtype=str)
    terms_df = pd.DataFrame({'terms': terms})
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    df_keys = train_df.keys()
    if 'annotations' in df_keys:
        annotations = train_df['annotations'].values
    else:
        annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))

    test_df = pd.read_pickle(test_data_file)

    print("Length of test set: " + str(len(test_df)))

    test_labels = np.zeros((len(test_df), len(terms)), dtype=np.int32)
    for i, row in enumerate(test_df.itertuples()):
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                test_labels[i, terms_dict[go_id]] = 1

    # Annotations
    test_annotations = []
    for i, row in enumerate(test_df.itertuples()):
        annots = set()
        for go_id in row.prop_annotations:
            if go.has_term(go_id):
                annots |= go.get_anchestors(go_id)
        test_annotations.append(annots)

    go.calculate_ic(annotations + test_annotations)

    # DeepGO
    go_set = go.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])

    labels = test_annotations
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))


    def get_metrics(predictions):
        fma = 0.0
        tmax = 0.0
        smin = 1000.0
        precisions = []
        recalls = []
        best_preds = []
        for t in range(101):
            threshold = t / 100.0
            preds = []
            for i, row in enumerate(test_df.itertuples()):
                annots = set()
                for j, score in enumerate(predictions[i]):
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
            if fma < fscore:
                fma = fscore
                tmax = threshold
                best_preds = preds
            if smin > s:
                smin = s

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]
        aupr = np.trapz(precisions, recalls)
        return fma,smin,aupr,tmax,best_preds




    fma,smin,aupr,tmax,best_preds = get_metrics(predictions)
    best_out = [list(row_out) for row_out in best_preds]


    if os.path.exists('Results.pkl'):
        best_out_df = pd.read_csv('Results.csv',index_col=0,)
        print(best_out_df.keys())
        best_out_df[f'{feats_type}'] = best_out
    else:
        best_out_df = pd.DataFrame()
        best_out_df[f'{feats_type}'] = best_out

    best_out_df = best_out_df.reset_index(drop=True)
    best_out_df.to_csv('Results.csv')

    print(f'Fmax: {fma:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}')
    F_txt.close()


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


if __name__ == '__main__':
    main()
