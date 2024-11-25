from test_function import *
import scarches as sca
import os
from GeneSymbolUniform_Pytoolkit.Pytoolkit_GeneSymbolUniform import GeneSymbolUniform

def cell_states_annot(
        model_dir,
        adata_test,
        finetune_epoch = 20,
        gene_uniform = True
):
    '''
    cell_states_annot
    This function is used to annotate liver cell types and atypical cell states (deviated / intermediate).
    Input:
        model_dir: path to the model, eg."./model"
        adata_test: anndata to annotate, 
                    note that initial counts matrix should be provided in X or layers['counts'],
                    and batch information can be provided in obs['batch']
        finetune_epoch: number of epochs for finetuning scArches model
    Output:
        pred_res: a dataframe of annotation results
        query_latent: adata object of latent representation obtained from scArches model
    '''
    ref_path = model_dir + "/scarches_model_new/"
    model_path = model_dir + "/LiverCT/"
    model_name = "latent_input_classifier_all_adjparam_unbalancedsvm_new"

    var_names = pd.read_csv(ref_path+"var_names.csv",header = None).iloc[:,0]
    if 'counts' in adata_test.layers.keys():
        adata_test.X = adata_test.layers['counts']
    meta = adata_test.obs

    # --------------------- gene alignment -----------------------------
    if gene_uniform:
        test = GeneSymbolUniform(input_adata = adata_test,
                                      ref_table_path = "./GeneSymbolUniform_Pytoolkit/GeneSymbolRef_SelectAll_upd.csv",
                                      gene_list_path = "./GeneSymbolUniform_Pytoolkit/total_gene_list_43878.txt",
                                      output_dir="./test_pytoolkit/",
                                      output_prefix='Lu2022',
                                      print_report=False,
                                      average_alias=False,
                                      n_threads=30)
        test = feature_alignment(test, var_names)
    else:
        test = feature_alignment(adata_test, var_names)
    
    if 'batch' in meta.columns:
        test.obs['batch'] = list(meta['batch'])
    else:
        test.obs['batch'] = "new_batch"
    test.obs['level1'] = "Unknown"  # unlabeled_category_
    test.layers['counts'] = test.X
    
    model = sca.models.SCANVI.load_query_data(
        test,
        ref_path,
        freeze_dropout = True,
    )
    model._unlabeled_indices = np.arange(test.n_obs)
    model._labeled_indices = []
    model.train(
        max_epochs=finetune_epoch,
        plan_kwargs=dict(weight_decay=0.0),
        check_val_every_n_epoch=10,
    )
    model.save(model_dir + "/my_surgery_model/", overwrite=True)
    query_latent = sc.AnnData(model.get_latent_representation())
    query_latent.obs_names = adata_test.obs_names
    pred_res = LabelPredict(query_latent, is_cnt=False, select_feature=False,
                            model_path=model_path, model_name=model_name, 
                            transition_predict=True, disease_predict=True)
    pred_res.index = test.obs_names
    return pred_res, query_latent


def hep_zonation_annot(
        model_dir,
        hepatocyte_adata,
        finetune_epoch = 20,
):
    '''
    hep_zonation_annot
    This function is used to annotate zonation labels of hepatocytes. 
    There are 3 zones: C (Central), M (Mid), P (Periportal+Portal).
    Input:
        model_dir: path to the model, eg."./model"
        hepatocyte_adata: adata object contains only hepatocytes, 
                            note that initial counts matrix should be provided in X or layers['counts'],
                            and donor_ID can be provided in obs['donor_ID'] for batch correction among donors
        finetune_epoch: number of epochs for finetuning scArches model
    Output:
        pred_res: a dataframe of zonation labels
        query_latent: adata object of latent representation obtained from scArches model
    '''
    ref_path = model_dir + "/hep_scanvi_finetune/"
    hep_clf_path = model_dir + "/LiverCT/hep_latent_rf_g3.pkl"

    var_names = pd.read_csv(ref_path+"var_names.csv",header = None).iloc[:,0]
    if 'counts' in hepatocyte_adata.layers.keys():
        hepatocyte_adata.X = hepatocyte_adata.layers['counts']
    meta = hepatocyte_adata.obs

    # --------------------- gene alignment -----------------------------
    test = GeneSymbolUniform(input_adata = hepatocyte_adata,
                                    ref_table_path = "./GeneSymbolUniform_Pytoolkit/GeneSymbolRef_SelectAll_upd.csv",
                                    gene_list_path = "./GeneSymbolUniform_Pytoolkit/total_gene_list_43878.txt",
                                    output_dir="./test_pytoolkit/",
                                    output_prefix='Lu2022',
                                    print_report=False,
                                    average_alias=False,
                                    n_threads=30)
    test = feature_alignment(test, var_names)

    if 'donor_ID' in meta.columns:
        test.obs['donor_ID'] = list(meta['donor_ID'])
    else:
        test.obs['donor_ID'] = "new_donor"
    test.obs['level1'] = "Unknown"  # unlabeled_category_
    test.layers['counts'] = test.X
    
    model = sca.models.SCANVI.load_query_data(
        test,
        ref_path,
        freeze_dropout = True,
    )
    model._unlabeled_indices = np.arange(test.n_obs)
    model._labeled_indices = []
    model.train(
        max_epochs=finetune_epoch,
        plan_kwargs=dict(weight_decay=0.0),
        check_val_every_n_epoch=10,
    )
    model.save(model_dir + "/my_surgery_model_hep/", overwrite=True)
    query_latent = sc.AnnData(model.get_latent_representation())
    query_latent.obs_names = hepatocyte_adata.obs_names

    clf = joblib.load(hep_clf_path)
    test.obs["zonation_pred"] = list(clf.predict(query_latent.X))
    return test.obs["zonation_pred"], query_latent
