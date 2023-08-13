# 230714 update: add disease/transition score; correct suggest_label_lineage
# 230715 update: debug Mig.cDC file name loading problem

# 230714 update: add disease/transition score; correct suggest_label_lineage

from scipy.sparse import csr_matrix
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import joblib
import datetime
import os
from functools import reduce

def feature_alignment(test_set, gene_list):
    # set test_set features as gene_list, zero-filling for missing features
    selected_set = set(gene_list)
    test_set_genes = set(test_set.var_names)
    common_set = selected_set & test_set_genes
    gene_extra = selected_set - common_set
    n_extra = len(gene_extra)
    if n_extra / len(gene_list) > 0.05:
        print("Warning: %d features not exist in testset." % len(gene_extra))

    if n_extra > 0:  # fill zeros for missing features
        # 直接扩充csr大小，储存元素不变，相当于zero-filling (hstack速度慢)
        new_mtx = csr_matrix(test_set.X, shape=(test_set.n_obs, test_set.n_vars + n_extra))
        test_adata = anndata.AnnData(new_mtx)
        test_adata.obs_names = test_set.obs_names
        test_adata.var_names = list(test_set.var_names) + list(gene_extra)
        # test_adata.obs = test_set.obs
        return test_adata[:, gene_list]
    else:
        return test_set[:, gene_list]


def LabelPredict(test_set, is_cnt, select_feature, 
                 model_path, model_name, transition_predict=True, disease_predict=True, 
                 skip_lv1=False, lv1_label=None):
    """
    :param test_set:
    :param is_cnt:
    :param save_path:
    :param model_name:
    :return:
    """

    # --------------------- feature selection --------------------------
    print(datetime.datetime.now(), "Test set feature selection start")
    # 如果没有选择基因，训练存储的gene_selected_all是trainset_gene_list
    if select_feature:
        gene_selected = pd.read_csv(model_path + "/" + model_name + "/train_gene_list.csv", index_col=0)
        gene_selected = list(gene_selected.values.T[0])
        test_adata = feature_alignment(test_set, gene_selected)
    else:
        test_adata = test_set
        gene_selected = test_adata.var_names

    # --------------------- normalization --------------------------
    if is_cnt:
        print(datetime.datetime.now(), "Test set normalization start")
        sc.pp.normalize_total(test_adata, target_sum=1e4)
        sc.pp.log1p(test_adata)
    else:
        print("Normalization has been escaped.")

    # --------------------- lv1 predict --------------------------
    if skip_lv1:
        test_adata.obs['pred_lv1'] = lv1_label
        test_adata.obs['proba_lv1'] = 1
    else:
        print(datetime.datetime.now(), "Test set lv1 predict start...")
        if select_feature:
            gene_selected_lv1 = pd.read_csv(model_path + "/" + model_name + "/gene_selected.csv", index_col=0)
            gene_selected_lv1 = list(gene_selected_lv1.values.T[0])
        else:
            gene_selected_lv1 = gene_selected
        clf = joblib.load(model_path + "/" + model_name + "/mlp_clf_lv1.pkl")
        proba_mtx_lv1 = clf.predict_proba(test_adata[:, gene_selected_lv1].X)
        test_adata.obs['pred_lv1'] = clf.predict(test_adata[:, gene_selected_lv1].X)
        test_adata.obs['proba_lv1'] = np.max(proba_mtx_lv1, axis=1)

    # --------------------- lv2 predict --------------------------
    print(datetime.datetime.now(), "Test set lv2 predict start...")
    adata_list = []
    celltype_tree = np.load(model_path + "/" + model_name + "/celltype_tree.npy", allow_pickle='TRUE').item()
    file_list = np.array(os.listdir (model_path + "/" + model_name + '/'))
    for g_label in np.unique(test_adata.obs['pred_lv1']):
        lv2_celltypes = celltype_tree[g_label]
        start = datetime.datetime.now()
        print(start, g_label, "lv2 predict...")
        if os.path.exists(model_path + "/" + model_name + '/voting_clf_lv2_' + g_label + '.pkl'):
            if select_feature:
                gene_selected_lv2 = pd.read_csv(model_path + "/" + model_name + "/gene_selected_" + g_label + ".csv",
                                                index_col=0)
                gene_selected_lv2 = list(gene_selected_lv2.values.T[0])
                gene_selected_oc = gene_selected_lv2
            else:
                gene_selected_lv2 = gene_selected
                gene_selected_oc = gene_selected

            adata_part = test_adata[test_adata.obs["pred_lv1"] == g_label]
            clf2 = joblib.load(model_path + "/" + model_name + '/voting_clf_lv2_' + g_label + '.pkl')
            proba = clf2.predict_proba(adata_part[:, gene_selected_lv2].X)
            proba_sort = np.sort(proba, axis=1)
            rank = np.argsort(proba, axis=1)

            adata_part.obs["voting_lv2_1"] = list(clf2.classes_[rank[:, -1]])
            adata_part.obs["voting_proba_1"] = list(adata_part.obs['proba_lv1'] * proba_sort[:, -1])
            adata_part.obs["voting_lv2_2"] = list(clf2.classes_[rank[:, -2]])
            adata_part.obs["voting_proba_2"] = list(adata_part.obs['proba_lv1'] * proba_sort[:, -2])

            # one vs one
            if transition_predict:
                pred_list = np.full(adata_part.n_obs, fill_value=None)
                distance_list = np.full(adata_part.n_obs, fill_value=None)
                dis_norm_list = np.full(adata_part.n_obs, fill_value=None)

                file_list_ovo = file_list[[i.split('_')[1] == 'ovo' for i in file_list]]
                file_list_ovo = file_list_ovo[[i.split('_')[0] == g_label for i in file_list_ovo]]
                for file in file_list_ovo:
                    lv2_label1 = file.split('_')[2]
                    lv2_label2 = file.split('_')[4].split('.pkl')[0]
                    slice1 = (adata_part.obs["voting_lv2_1"] == lv2_label1) & (adata_part.obs["voting_lv2_2"] == lv2_label2)
                    slice2 = (adata_part.obs["voting_lv2_1"] == lv2_label2) & (adata_part.obs["voting_lv2_2"] == lv2_label1)
                    cell_slice = slice1 | slice2
                    if sum(cell_slice) > 0:
                        ovo_clf = joblib.load(model_path + "/" + model_name + '/' + file)
                        gene_selected_ovo = gene_selected_lv2
                        pred_list[cell_slice] = ovo_clf.predict(adata_part[cell_slice, gene_selected_ovo].X)
                        distance = abs(ovo_clf.decision_function(adata_part[cell_slice, gene_selected_ovo].X))
                        distance_list[cell_slice] = distance
                        dis_norm_list[cell_slice] = distance / np.percentile(distance, 80)
                adata_part.obs["pred_lv2_ovo"] = list(pred_list)
                adata_part.obs["distance_ovo"] = list(distance_list)
                adata_part.obs["distance_norm_ovo"] = list(dis_norm_list)

        else:  # lv1_celltype only conclude one lv2_celltype
            gene_selected_oc = gene_selected
            adata_part = test_adata[test_adata.obs["pred_lv1"] == g_label, :]
            adata_part.obs["voting_lv2_1"] = celltype_tree[g_label][0]  # adata_part.obs['pred_lv1']
            adata_part.obs["voting_proba_1"] = list(adata_part.obs['proba_lv1'])
            adata_part.obs["voting_lv2_2"] = ["Unclassified"] * adata_part.n_obs
            adata_part.obs["voting_proba_2"] = [0] * adata_part.n_obs
            if transition_predict:
                adata_part.obs["pred_lv2_ovo"] = adata_part.obs['pred_lv1']
                adata_part.obs["distance_ovo"] = [1e4] * adata_part.n_obs
                adata_part.obs["distance_norm_ovo"] = [1] * adata_part.n_obs  #设为最大值1
        
        # one class
        if disease_predict:
            file_list_oc = file_list[[i.split('_')[1] == 'oneclass' for i in file_list]]
            is_normal = np.full(adata_part.n_obs, fill_value=None)
            oc_distance = np.full(adata_part.n_obs, fill_value=None)
            oc_dis_norm = np.full(adata_part.n_obs, fill_value=None)
            for lv2_label in lv2_celltypes:
                cell_slice = adata_part.obs['voting_lv2_1'] == lv2_label
                if sum(cell_slice) > 0:
                    adata_oc = adata_part[cell_slice, gene_selected_oc]
                    file_list_oc_use = file_list_oc[[i.split('_')[2] == lv2_label for i in file_list_oc]]
                    # file_list_oc_use = file_list_oc_use[[i.split('_')[2] == str(nu) for i in file_list_oc_use]]
                    if len(file_list_oc_use) > 1:
                        print("########### Warning: More than one one-class pkgs for", lv2_label)
                    file = file_list_oc_use[0]
                    # nu = file.split('_')[3]
                    # gamma = file.split('_')[4]
                    oc_clf = joblib.load(model_path + "/" + model_name + '/' + file)
                    is_normal[cell_slice] = oc_clf.predict(adata_oc.X)
                    distance = oc_clf.decision_function(adata_oc.X)
                    oc_distance[cell_slice] = distance
                    oc_dis_norm[cell_slice] = distance / np.percentile(abs(distance), 80)
            adata_part.obs["is_normal"] = list(is_normal)
            adata_part.obs["oc_distance"] = list(oc_distance)
            adata_part.obs["oc_score"] = list(oc_dis_norm)

        adata_list.append(adata_part)
        end = datetime.datetime.now()
        time = (end - start).total_seconds()
        print(g_label + " predict is finished in", time)

    test_adata_labeled = anndata.concat(adata_list, merge="same")
    test_adata = test_adata_labeled[test_adata.obs_names, :]
    
    # # 可以把None改成"Unknown"
    # label_lv1 = np.full(test_adata.n_obs, fill_value=None)
    # label_lv2 = np.full(test_adata.n_obs, fill_value=None)
    
    # threshold = 0.6
    # label_lv1[test_adata.obs['proba_lv1'] > threshold] = test_adata.obs['pred_lv1'][test_adata.obs['proba_lv1'] > threshold]
    # label_lv2[test_adata.obs['proba_lv1'] > threshold] = test_adata.obs['voting_lv2_1'][test_adata.obs['proba_lv1'] > threshold]
    # epi = (test_adata.obs['pred_lv1']=='Hepatocyte') | (test_adata.obs['pred_lv1']=='Cholangiocyte')
    # unknown_epi = epi & (test_adata.obs['proba_lv1'] <= threshold)
    # label_lv1[unknown_epi] = "UnknownEpithelial"
    # label_lv2[unknown_epi] = "UnknownEpithelial"
    # test_adata.obs['suggest_label_lv1'] = list(label_lv1)
    # test_adata.obs['suggest_label_lv2'] = list(label_lv2)

    # lineage = np.array(test_adata.obs['suggest_label_lv1'])
    # lineage[(test_adata.obs['suggest_label_lv1']=='TNK cell') | (test_adata.obs['suggest_label_lv1']=='B cell')
    #          | (test_adata.obs['suggest_label_lv1']=='Plasma B cell')] = 'Lymphoid'
    # lineage[test_adata.obs['suggest_label_lv1']=='Myeloid cell'] = 'Myeloid'
    # lineage[test_adata.obs['suggest_label_lv1']=='Mesenchymal cell'] = 'Stromal'
    # lineage[test_adata.obs['suggest_label_lv1']=='Endothelial cell'] = 'Endothelial'
    # lineage[(test_adata.obs['suggest_label_lv1']=='Hepatocyte') | (test_adata.obs['suggest_label_lv1']=='Cholangiocyte')
    #         | (test_adata.obs['suggest_label_lv1']=='UnknownEpithelial')] = 'Epithelial'
    # test_adata.obs['suggest_label_lineage'] = list(lineage)
    lineage = np.array(test_adata.obs['pred_lv1'])
    lineage[(test_adata.obs['pred_lv1']=='TNK cell') | (test_adata.obs['pred_lv1']=='B cell')
             | (test_adata.obs['pred_lv1']=='Plasma B cell')] = 'Lymphoid'
    lineage[test_adata.obs['pred_lv1']=='Myeloid cell'] = 'Myeloid'
    lineage[test_adata.obs['pred_lv1']=='Mesenchymal cell'] = 'Stromal'
    lineage[test_adata.obs['pred_lv1']=='Endothelial cell'] = 'Endothelial'
    lineage[(test_adata.obs['pred_lv1']=='Hepatocyte') | (test_adata.obs['pred_lv1']=='Cholangiocyte')] = 'Epithelial'
    test_adata.obs['pred_lineage'] = list(lineage)
    test_adata.obs['pred_lv2'] = list(test_adata.obs['voting_lv2_1'])


    # disease score / transition score
    oc_score = test_adata.obs['oc_score']
    ovo_score = test_adata.obs['distance_norm_ovo']
    oc_score[oc_score > 1] = 1
    oc_score[oc_score < -1] = -1

    ovo_score[ovo_score > 1] = 1
    test_adata.obs['disease_distance'] = list(oc_score)
    test_adata.obs['transition_distance'] = list(ovo_score)
    
    # exp score
    test_adata.obs["intermediate_score"] = np.exp(-test_adata.obs['distance_ovo'])
    # disease score
    test_adata.obs["deviated_score"] = -test_adata.obs['disease_distance']

    # epi intermediate
    intermediate_score = np.array(test_adata.obs["intermediate_score"])
    cell_slice = (test_adata.obs['pred_lineage'] == 'Epithelial')
    svm_clf = joblib.load(model_path + "/" + model_name + "/epi_intermediate_svm.pkl")
    distance_origin = abs(svm_clf.decision_function(test_adata[cell_slice, gene_selected_ovo].X))
    scores = np.exp(-distance_origin)
    intermediate_score[cell_slice] = list(scores)
    test_adata.obs["intermediate_score"] = list(intermediate_score)


    # annotation type
    test_adata.obs["intermediate_state"] = ["intermediate" if i > 0.2 else "non-intermediate" for i in test_adata.obs[ 'intermediate_score']]
    test_adata.obs["deviated_state"] = ["deviated" if i > 0 else "non-deviated" for i in test_adata.obs[ 'deviated_score']]
    
    df_groups = [test_adata.obs["pred_lineage"], test_adata.obs["pred_lv1"], test_adata.obs["pred_lv2"],
                 test_adata.obs["intermediate_score"], test_adata.obs['deviated_score'], 
                 test_adata.obs["intermediate_state"], test_adata.obs["deviated_state"], 
                 test_adata.obs["proba_lv1"], test_adata.obs["voting_lv2_1"], test_adata.obs["voting_lv2_2"]]
    softvote_pred = reduce(lambda left, right: pd.concat([left, right], axis=1), df_groups)
    return softvote_pred
