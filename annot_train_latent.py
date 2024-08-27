from train_function import *
from test_function import *
import scarches as sca
import sys
model_name = sys.argv[1]

save_path = "../../model"
# model_name = "latent_input_classifier_all_adjparam_new"
ref_path = '../../model/scarches_model_intersect/'

# adata_train = sc.read_h5ad("../../data/normal_atlas/43878_new/normal_annot_43878_new.h5ad")
adata_train = sc.read_h5ad("/home/wyh/liver_atlas/data/normal_atlas/adata_healthy_intersect_counts.h5ad")
if "counts" not in adata_train.layers.keys():
    adata_train.layers["counts"] = adata_train.X

var_names = pd.read_csv(ref_path+"/var_names.csv",header = None).iloc[:,0]
adata_train = feature_alignment(adata_train, var_names)

model = sca.models.SCANVI.load_query_data(
    adata_train,
    ref_path,
    freeze_dropout = True,
)
model.train(
    max_epochs=0,
    plan_kwargs=dict(weight_decay=0.0),
    check_val_every_n_epoch=10,
)
latent_train = sc.AnnData(model.get_latent_representation())
latent_train.obs = adata_train.obs

myClassifier_train(latent_train, is_cnt=False, select_feature=False,
                   gpu_available=True,
                   label_lv1='level1', label_lv2='level2',
#                    label_lv1='annot_lv1', label_lv2='annot_lv2',
                   model_name=model_name, save_path=save_path)