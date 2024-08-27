import scanpy as sc
import anndata as ad
import scarches as sca
import datetime

# train_set = sc.read_h5ad("../../data/normal_atlas/43878_new/normal_annot_43878_new.h5ad")
train_set = sc.read_h5ad("/home/wyh/liver_atlas/data/normal_atlas/adata_healthy_intersect_counts.h5ad")
if "counts" not in train_set.layers.keys():
    train_set.layers["counts"] = train_set.X

def scanvi_train(adata):
    start=datetime.datetime.now()
    adata.raw = adata
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=2000,
        layer="counts",
        batch_key="batch",
        subset=True
    )
    sca.models.SCVI.setup_anndata(adata, layer="counts", batch_key="batch", labels_key="level1")
    # (adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    vae = sca.models.SCVI(
        adata,
        n_layers=2,
        n_latent=30,
        gene_likelihood="nb",
        encode_covariates=True,
        deeply_inject_covariates=False,
        use_layer_norm="both",
        use_batch_norm="none",
    )
    vae.train()

    lvae = sca.models.SCANVI.from_scvi_model(
        vae,
        unlabeled_category="Unknown",
    )
    lvae.train(max_epochs=20, n_samples_per_label=100)
    ref_path = '../../model/scarches_model_intersect/'
    end=datetime.datetime.now()
    lvae.save(ref_path, overwrite=True)
    print("scanvi train(Seconds):", end-start)
    return lvae

lvae = scanvi_train(adata=train_set)