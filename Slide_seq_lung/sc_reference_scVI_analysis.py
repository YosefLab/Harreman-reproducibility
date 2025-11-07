import os
import scanpy as sc
import scvi

BASE_PATH = "/home/projects/nyosef/oier/Harreman_files/Slide_seq_lung"
ADATA_PATH = os.path.join(BASE_PATH, 'h5ads')
MODELS_PATH = os.path.join(BASE_PATH, 'models')

SAMPLE_ID_KEY = "donor_id"
SCVI_LATENT_KEY = "X_scVI"

N_TOP_GENES = 4000


adata = sc.read_h5ad(os.path.join(ADATA_PATH, 'Slide_seq_lung_sc_ref_adata.h5ad'))


def scVI_pipeline(adata, MODEL_PATH, DATA_PATH):

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=N_TOP_GENES,
        layer="counts",
        flavor="seurat_v3",
        batch_key=SAMPLE_ID_KEY,
        span=1.0
    )

    scvi_adata = adata[:, adata.var.highly_variable].copy()

    scvi.model.SCVI.setup_anndata(scvi_adata, layer="counts", batch_key=SAMPLE_ID_KEY)

    model = scvi.model.SCVI(scvi_adata)
    model.train(devices='1')
    model.save(MODEL_PATH, overwrite=True)

    model = scvi.model.SCVI.load(MODEL_PATH, scvi_adata)
    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

    sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY, n_neighbors=30)
    sc.tl.umap(adata)

    adata.write(DATA_PATH)


MODEL_PATH = os.path.join(MODELS_PATH, 'Slide_seq_lung_sc_ref_scVI')
DATA_PATH = os.path.join(ADATA_PATH, 'Slide_seq_lung_sc_ref_scVI_adata.h5ad')

print('Running scVI')

scVI_pipeline(adata, MODEL_PATH, DATA_PATH)
