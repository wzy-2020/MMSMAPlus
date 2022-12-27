import numpy as np
import pandas as pd
import click as ck
import math
import logging
import warnings
from utils import FUNC_DICT, Ontology, NAMESPACES
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
    '--ont', '-o', default='bp',
    help='GO subontology (bp, mf, cc)')
def main(root_data_dir, datasets_name, ont):
    # Load files
    net_type = 'MVPF_MSRA'
    feats_list = ['bert', 'pssm', 'onehot', 'opf']
    alphas =[0.7, 0.3, 0.0, 0.0]
    go_rels = Ontology(root_data_dir + datasets_name+'/go.obo', with_rels=True)
    train_data_file = f'data/{datasets_name}/train_data.pkl'
    test_data_file = f'data/{datasets_name}/{ont}/test_data.pkl'
    pred_data_file = f'models/{datasets_name}/{ont.upper()}/{net_type}/B_P_O/prediction.pkl'

    terms_file = f'data/{datasets_name}/{ont}/{ont}.txt'
    F_txt = open(f'models/{datasets_name}/{ont.upper()}/{ont.upper()}_DeepGOPlus_result.txt', 'a+')
    diamond_scores_file = f'data/{datasets_name}/test_diamond.res'
    terms = np.loadtxt(terms_file,dtype=str)
    terms_dict = {v: i for i, v in enumerate(terms)}
    terms_df = pd.DataFrame({'terms': terms})
    terms = terms_df['terms'].values.flatten()

    # single
    # predictions = pickle.load(open(pred_data_file,'rb'))['y_pred']
    # multi-view
    Output_list = []
    for name in feats_list:
        preds_file = f'models/cafa3/{ont}/{net_type}/{name}/prediction.pkl'
        test_df = pd.read_pickle(preds_file)
        output = np.array(test_df['preds'].tolist())
        Output_list.append(output)


    final_preds = 0.
    for id, (t_a, t_pred) in enumerate(zip(alphas, Output_list)):
        final_preds += t_a * t_pred

    final_preds = final_preds[98, :]
    final_preds = final_preds.reshape(1, 3992)

    train_df = pd.read_pickle(train_data_file)
    test_df = pd.read_pickle(test_data_file)
    test_df = test_df.loc[test_df['proteins'] == 'T100900008300']
    print("Length of test set: " + str(len(test_df)))
    print("Length of terms: " + str(len(terms)))
    annotations = train_df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = []
    for i, row in enumerate(test_df.itertuples()):
        annots = set()
        for go_id in row.annotations:
            if go_rels.has_term(go_id):
                annots |= go_rels.get_anchestors(go_id)
        test_annotations.append(annots)

    go_rels.calculate_ic(annotations + test_annotations)
    # Print IC values of terms
    ics = {}
    for term in terms:
        ics[term] = go_rels.get_ic(term)

    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i


    # BLAST Similarity (Diamond)
    diamond_scores = {}
    with open(diamond_scores_file) as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in diamond_scores:
                diamond_scores[it[0]] = {}
            diamond_scores[it[0]][it[1]] = float(it[2])

    blast_preds = []
    for i, row in enumerate(test_df.itertuples()):
        annots = {}
        prot_id = row.proteins
        # BlastKNN
        if prot_id in diamond_scores:
            sim_prots = diamond_scores[prot_id]
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                if p_id in prot_index:
                    allgos |= annotations[prot_index[p_id]]
                    total_score += score
            allgos = list(sorted(allgos))
            sim = np.zeros(len(allgos), dtype=np.float32)
            for j, go_id in enumerate(allgos):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if p_id in prot_index:
                        if go_id in annotations[prot_index[p_id]]:
                            s += score
                sim[j] = s / total_score
            ind = np.argsort(-sim)
            for go_id, score in zip(allgos, sim):
                annots[go_id] = score
        blast_preds.append(annots)

    # DeepGOPlus
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])

    labels = test_annotations
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
    pred_df = pd.DataFrame(columns=['proteins', 'y_true'])
    pred_df['proteins'] = test_df['proteins']
    pred_df['y_true'] = labels
    # print(len(go_set))
    deep_preds = []
    alphas = {NAMESPACES['mf']: 0.55, NAMESPACES['bp']: 0.59, NAMESPACES['cc']: 0.46}
    print(alphas,file=F_txt)

    for i, row in enumerate(test_df.itertuples()):
        annots_dict = blast_preds[i].copy()
        if bool(annots_dict):
            for go_id in annots_dict:
                if go_rels.get_term(go_id):
                    annots_dict[go_id] *= alphas[go_rels.get_namespace(go_id)]
            for j, score in enumerate(final_preds[i]):
                go_id = terms[j]
                score *= 1 - alphas[go_rels.get_namespace(go_id)]
                if go_id in annots_dict:
                    annots_dict[go_id] += score
                else:
                    annots_dict[go_id] = score

        else:
            for j, score in enumerate(final_preds[i]):
                go_id = terms[j]
                annots_dict[go_id] = score

        deep_preds.append(annots_dict)


    fmax = 0.0
    tmax = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    rus = []
    mis = []
    best_preds = None
    for t in range(1, 101):  # the range in this loop has influence in the AUPR output
        threshold = t / 100.0
        preds = []
        for i, row in enumerate(test_df.itertuples()):
            annots = set()
            for go_id, score in deep_preds[i].items():
                if score >= threshold:
                    annots.add(go_id)

            new_annots = set()
            for go_id in annots:
                new_annots |= go_rels.get_anchestors(go_id)
            preds.append(new_annots)

        # Filter classes
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))

        fscore, prec, rec, s, ru, mi, fps, fns = evaluate_annotations(go_rels, labels, preds)
        avg_fp = sum(map(lambda x: len(x), fps)) / len(fps)
        avg_ic = sum(map(lambda x: sum(map(lambda go_id: go_rels.get_ic(go_id), x)), fps)) / len(fps)
        # print(f'{avg_fp} {avg_ic}')
        precisions.append(prec)
        recalls.append(rec)
        print(f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} S: {s}, RU: {ru}, MI: {mi} threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
            best_preds = preds
        if smin > s:
            smin = s
    best_preds = best_preds[0]
    truth = test_df['annotations'].tolist()[0]
    pred_truth = list(best_preds & truth)
    best_preds = list(best_preds)
    best_preds.sort()

    print(' '.join(best_preds))
    pred_truth.sort()
    print(pred_truth)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'datasets:{datasets_name}\t Ont:{ont}')
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}')
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}', file=F_txt)
    # pred_df.to_pickle(f'case_study/DeepGOPlus_pred.pkl')
    F_txt.close()



def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
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
    return f, p, r, s, ru, mi, fps, fns

if __name__ == '__main__':
    main()