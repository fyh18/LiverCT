from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier
import scipy
import numpy as np
import pandas as pd
import scanpy as sc
import joblib
import datetime
import os


def feature_select(adata, y, file_name):
    print("Features selection start...")
    anova_filter = SelectKBest(f_classif, k=2000)
    anova_filter.fit(adata.X, y)
    mask_selected = anova_filter.get_support(indices=False)
    gene_selected = adata.var_names[mask_selected]
    if file_name != None:
        pd.DataFrame(gene_selected).to_csv(file_name + '.csv')
    print("finished")
    return gene_selected


def voting_clf_train(X, y, file_name, gpu_available):
    print("Voting clf train start...")
    if gpu_available:
        tree_method = "gpu_hist"
    else:
        tree_method = "hist"
    # 多核并行 use all cpu kernels
    with joblib.parallel_backend('threading', n_jobs=-1):
        start = datetime.datetime.now()
        rf_clf = RandomForestClassifier(n_estimators=50, random_state=1)
        mlp_clf = MLPClassifier(hidden_layer_sizes=[128, 64, 32], random_state=1, max_iter=500)
        ovr_lr_clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        xgb_clf = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=3,
                                objective="multi:softprob", num_class=len(np.unique(y)),
                                tree_method=tree_method)
        voting_clf = VotingClassifier(estimators=[('rf', rf_clf), ('mlp', mlp_clf),
                                                  ('ovr_lr', ovr_lr_clf), ('xgb', xgb_clf)],
                                      voting='soft')
        voting_clf.fit(X, y)
        if file_name is not None:
            joblib.dump(voting_clf, file_name + '.pkl')
        end = datetime.datetime.now()
        time = (end - start).total_seconds()
    print("Voting clf train finished. Training time: %s Seconds" % time)


def ovo_svm_train(X, y, file_name, label1, label2):
    # 用于判定transition_state，只用于最大可能的两类
    print("SVM train start...")
    start = datetime.datetime.now()
    # smaller C allows softer margin
    # svm_clf = SVC(C=0.1, kernel="linear", class_weight='balanced',
    #                random_state=0, tol=1e-03, max_iter=1000)
    
    svm_clf_sgd = SGDClassifier(loss="hinge", #class_weight='balanced', 
                                alpha=0.01, random_state=0, tol=1e-04, max_iter=5000)
    svm_clf = make_pipeline(StandardScaler(), svm_clf_sgd)
    svm_clf.fit(X, y)
    end = datetime.datetime.now()
    time = (end - start).total_seconds()
    if file_name is not None:
        joblib.dump(svm_clf, file_name + '_ovo_' + label1 + '_vs_' + label2 + '.pkl')
    print(label1 + '_vs_' + label2 + "_SVM train finished. Training time: %s Seconds" % (end - start))
    return time


def oneclass_svm_train(X, file_name, label, nu, gamma):
    # for outlier detection
    print("SVM train start...")
    start = datetime.datetime.now()
    # clf = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
    # SGDOneClassSVM is well suited for datasets with a large number of training samples
    # https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgdocsvm_vs_ocsvm.html
    transform = Nystroem(gamma=gamma, random_state=0)
    svm_clf = SGDOneClassSVM(nu=nu, shuffle=True, fit_intercept=True, random_state=0, tol=1e-04, max_iter=5000)
    clf = make_pipeline(transform, svm_clf)
    clf.fit(X)
    end = datetime.datetime.now()
    time = (end - start).total_seconds()
    if file_name is not None:
        joblib.dump(clf, file_name + '_oneclass_' + str(nu) + '_' + str(gamma) + '_' + label + '.pkl')
    print(label + str(nu) + str(gamma) + "_OneClassSVM train finished. Training time: %s Seconds" % (end - start))
    return clf


def myClassifier_train(adata_train, is_cnt, label_lv1, label_lv2, select_feature,
                       gpu_available, model_name, save_path):
    if not os.path.exists(save_path + "/saves/" + model_name):
        os.makedirs(save_path + "/saves/" + model_name)
    pd.DataFrame(adata_train.var_names).to_csv(save_path + "/saves/" + model_name + "/train_gene_list.csv")
    # preprocessing
    adata_train.var_names_make_unique()
    if is_cnt:
        print(datetime.datetime.now(), "Normalization start")
        sc.pp.normalize_total(adata_train, target_sum=1e4)
        sc.pp.log1p(adata_train)
    else:
        print(datetime.datetime.now(), "Normalization is skipped")

    if select_feature:
        print(datetime.datetime.now(), "Gene selection for lv1 start")
        gene_selected_all = set()
        gene_selected = feature_select(adata=adata_train, y=adata_train.obs[label_lv1],
                                       file_name=save_path + "/saves/" + model_name + "/gene_selected")
        gene_selected_all = gene_selected_all | set(gene_selected)
    else:
        gene_selected = adata_train.var_names

    # train
    # Level 1
    # ! lv1也换成了voting old:mlp_train
    voting_clf_train(X=adata_train[:, gene_selected].X, y=adata_train.obs[label_lv1],
                     gpu_available=gpu_available,
                     file_name=save_path + "/saves/" + model_name + "/mlp_clf_lv1")

    # Level 2
    # mlp
    celltype_tree = dict()
    lv1_celltype_list = np.unique(adata_train.obs[label_lv1])
    for g_label in lv1_celltype_list:
        adata_part = adata_train[adata_train.obs[label_lv1] == g_label, :]
        lv2_celltypes = np.unique(adata_part.obs[label_lv2])
        celltype_tree[g_label] = lv2_celltypes

        # 特征选择
        if select_feature:
            gene_selected = feature_select(adata=adata_part, y=adata_part.obs[label_lv2],
                                           file_name=save_path + "/saves/" + model_name + "/gene_selected_" + g_label)
            gene_selected_all = gene_selected_all | set(gene_selected)

        # 集成分类器训练
        if len(np.unique(adata_part.obs[label_lv2])) > 1:
            print("----------" + g_label + "----------")
            voting_clf_train(X=adata_part[:, gene_selected].X, y=adata_part.obs[label_lv2],
                             gpu_available=gpu_available,
                             file_name=save_path + "/saves/" + model_name + "/voting_clf_lv2_" + g_label)

        # one-class and one-vs-one svm
        for i in range(len(lv2_celltypes)):
            lv2_label1 = lv2_celltypes[i]
            # one-class
            X_oc = adata_part[adata_part.obs[label_lv2] == lv2_label1, gene_selected].X
            if scipy.sparse.issparse(X_oc):
                gamma = 1 / (len(gene_selected) * np.var(X_oc.toarray()))
            else:
                gamma = 1 / (len(gene_selected) * np.var(X_oc))
            for nu in np.arange(0.01, 0.2, 0.01):
                # for gamma in [1e-5, 1e-4, 1e-3, 1e-2]:
                oc_clf = oneclass_svm_train(X=X_oc,
                             file_name=None,
                             label=lv2_label1, nu=nu, gamma=gamma)
                res = oc_clf.predict(X_oc)
                if sum(res==-1) / len(res) > 0.1:
                    break
            joblib.dump(oc_clf, save_path + "/saves/" + model_name + "/"
                                + '_oneclass_' + lv2_label1 + '_' 
                                + str(nu) + '_' + str(gamma) + '.pkl')

            # one-vs-one
            for j in range(i+1, len(lv2_celltypes)):
                lv2_label2 = lv2_celltypes[j]
                adata_part_ovo = adata_part[(adata_part.obs[label_lv2] == lv2_label1) |
                                            (adata_part.obs[label_lv2] == lv2_label2)]
#                 if select_feature:
#                     # 挑两类之间的基因
#                     gene_selected = feature_select(adata=adata_part_ovo, y=adata_part_ovo.obs[label_lv2],
#                                          file_name=save_path + "/saves/" + model_name +
#                                                    "/gene_selected_" + lv2_label1 + lv2_label2)
#                     gene_selected_all = gene_selected_all | set(gene_selected)
                ovo_svm_train(X=adata_part_ovo[:, gene_selected].X, y=adata_part_ovo.obs[label_lv2],
                              file_name=save_path + "/saves/" + model_name + "/"+ g_label,
                              label1=lv2_label1, label2=lv2_label2)

    if select_feature:
        pd.DataFrame(gene_selected_all).to_csv(save_path + "/saves/" + model_name + "/gene_selected_all.csv")
    else:
        pd.DataFrame(adata_train.obs_names).to_csv(save_path + "/saves/" + model_name + "/gene_selected_all.csv")

    np.save(save_path + "/saves/" + model_name + "/celltype_tree.npy", celltype_tree)

