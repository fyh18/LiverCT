from test_function import *
import scarches as sca
import sys
import os
test_file_path = sys.argv[1]
test_file_name = sys.argv[2]
finetune_epoch = int(sys.argv[3])

test = sc.read_h5ad(test_file_path + test_file_name + ".h5ad")
meta = test.obs

# scarches
ref_path = '../../model/scarches_model_new/'
# model存放路径:
# model_path/saves/model_name/*
model_path = "../../model"
model_name = "latent_input_classifier_all_adjparam_unbalancedsvm_new"

var_names = pd.read_csv(ref_path+"var_names.csv",header = None).iloc[:,0]
test = feature_alignment(test, var_names)
test.obs['batch'] = list(meta['batch'])
test.obs['level1'] = "Unknown"  # unlabeled_category_
if 'counts' not in test.layers.keys():
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



def mkdir(path):
    folder = os.path.exists(path)

    if not folder: 
        os.makedirs(path) 
        print("--- new folder... ---")
        print("--- OK ---")
    else:
        print("--- There is this folder! ---")
    

if finetune_epoch!=0:
    ref_path = test_file_path+"/model_0806"
    mkdir(ref_path) #调用函数
    model.save(ref_path, overwrite=True)

query_latent = sc.AnnData(model.get_latent_representation())
query_latent.write(test_file_path + test_file_name + "_latent.h5ad")

# adata.X是原始counts
pred_res = LabelPredict(query_latent, is_cnt=False, select_feature=False,
                            model_path=model_path, model_name=model_name, 
                            transition_predict=True, disease_predict=True)
pred_res.index = test.obs_names
mkdir("./celltype0806/")
pred_res.to_csv("./celltype0806/" + test_file_name + "_epoch" + str(finetune_epoch) + "_annot_results.csv")
