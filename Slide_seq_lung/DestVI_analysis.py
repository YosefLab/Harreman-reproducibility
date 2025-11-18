import sys
sys.path.insert(0, '/home/projects/nyosef/oier/new_destvi_updated/scvi-tools/src')

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scvi.model import CondSCVI, DestVI


BASE_PATH = "/home/projects/nyosef/oier/Harreman_files/Slide_seq_lung"
ADATA_PATH = os.path.join(BASE_PATH, 'h5ads')
MODELS_PATH = os.path.join(BASE_PATH, 'models')
SLIDE_SEQ_LUNG_DESTVI_ADATA_PATH = os.path.join(ADATA_PATH, 'Slide_seq_lung_DestVI_v2_adata.h5ad')
SLIDE_SEQ_LUNG_SC_MODEL_PATH = os.path.join(MODELS_PATH, 'Slide_seq_lung_DestVI_v2_sc')
SLIDE_SEQ_LUNG_ST_MODEL_PATH = os.path.join(MODELS_PATH, 'Slide_seq_lung_DestVI_v2_st')

BATCH_KEY = "donor_id"
G = 8000
MIN_COUNTS = 10
CELL_TYPE_ID = "cell_type_coarse"
CELL_TYPE_HIGHRES_ID = "FinalCellType"


# If we want to use the DestVI output for cell-cell communication
TRANSPORTER_DB_PATH = '/home/projects/nyosef/oier/Harreman/data/HarremanDB/HarremanDB_human_extracellular.csv'
CELLCHATDB_PATH = '/home/projects/nyosef/oier/Harreman/data/CellChatDB'

database_info = {
    'mouse': {
        'interaction': os.path.join(CELLCHATDB_PATH, 'interaction_input_CellChatDB_v2_mouse.csv'),
        'complex': os.path.join(CELLCHATDB_PATH, 'complex_input_CellChatDB_v2_mouse.csv'),
    },
    'human': {
        'interaction': os.path.join(CELLCHATDB_PATH, 'interaction_input_CellChatDB_v2_human.csv'),
        'complex': os.path.join(CELLCHATDB_PATH, 'complex_input_CellChatDB_v2_human.csv'),
    }
}

interaction = pd.read_csv(database_info['human']['interaction'], index_col=0)
complex = pd.read_csv(database_info['human']['complex'], header=0, index_col=0)

interaction = interaction.sort_values('annotation')
ligand = interaction.ligand.values
receptor = interaction.receptor.values
interaction.pop('ligand')
interaction.pop('receptor')

# Load the spatial data
st_adata = sc.read_h5ad(os.path.join(ADATA_PATH, 'Slide_seq_lung_adata.h5ad'))

for i in range(len(ligand)):
    for n in [ligand, receptor]:
        l = n[i]
        if l in complex.index:
            n[i] = complex.loc[l].dropna().values[pd.Series(complex.loc[l].dropna().values).isin(st_adata.var_names)]
        else:
            n[i] = pd.Series(l).values[pd.Series(l).isin(st_adata.var_names)]

ligands = list(np.unique([s for sublist in ligand for s in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]))
receptors = list(np.unique([s for sublist in receptor for s in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]))

transporter_db_df = pd.read_csv(TRANSPORTER_DB_PATH, index_col=0)

all_genes = []

for i in range(len(transporter_db_df)):
    genes_split = transporter_db_df.loc[i, 'Gene'].split("/")
    genes_split = [gene.strip() for gene in genes_split]
    all_genes.append(genes_split)

transporters = [gene for genes in all_genes for gene in genes]
transporters = list(np.unique(transporters))

DB_genes = list(set(ligands) | set(receptors) | set(transporters))


# Load the single-cell data

sc_adata = sc.read_h5ad(os.path.join(ADATA_PATH, 'Slide_seq_lung_sc_ref_adata.h5ad'))

sc.pp.filter_genes(sc_adata, min_counts=MIN_COUNTS)

sc.pp.highly_variable_genes(
    sc_adata, n_top_genes=G, subset=False, layer="counts", flavor="seurat_v3"
)
DB_genes = list(set(DB_genes) & set(sc_adata.var_names))
sc_adata.var.loc[DB_genes, 'highly_variable'] = True
sc_adata = sc_adata[:,sc_adata.var['highly_variable']].copy()


# filter genes to be the same on the spatial data
intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
st_adata = st_adata[:, intersect].copy()
sc_adata = sc_adata[:, intersect].copy()
G = len(intersect)


# Fit the scLVM

train_sc = True
if train_sc:
    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key=CELL_TYPE_ID, fine_labels_key=CELL_TYPE_HIGHRES_ID, batch_key=BATCH_KEY)
    sc_model = CondSCVI(sc_adata, weight_obs=False, prior='mog', num_classes_mog=10)
    sc_model.train(batch_size=1024, max_epochs=2000)
    sc_model.save(SLIDE_SEQ_LUNG_SC_MODEL_PATH, overwrite=True, save_anndata=True)
else:
    sc_model = scvi.model.CondSCVI.load(SLIDE_SEQ_LUNG_SC_MODEL_PATH)


# Deconvolution with stLVM

st_adata = st_adata[st_adata.layers['counts'].sum(1)>10].copy()
st_adata.obs[BATCH_KEY] = 'spatial'

def spatial_nn_gex_smth(stadata,n_neighs):
    sc.pp.neighbors(stadata,n_neighs,use_rep="spatial",key_added="Xspatial")
    stadata.obsp['Xspatial_connectivities'] = stadata.obsp['Xspatial_connectivities'].ceil()
    stadata.obsp['Xspatial_connectivities'].setdiag(1)
    return stadata.obsp['Xspatial_connectivities'].dot(stadata.layers["counts"])
st_adata.layers["smoothed"]=spatial_nn_gex_smth(st_adata, n_neighs=5)

train_st = True
if train_st:
    st_model = DestVI.from_rna_model(st_adata, sc_model, add_celltypes=2, n_latent_amortization=None, anndata_setup_kwargs={'smoothed_layer': 'smoothed'}) # prior_mode = 'normal'
    st_model.train(max_epochs=1000, n_epochs_kl_warmup=200, batch_size=1024, plan_kwargs={'weighting_kl_latent': 1e-2, 'ct_sparsity_weight': 0})
    st_model.save(SLIDE_SEQ_LUNG_ST_MODEL_PATH, overwrite=True)
else:
    st_model = DestVI.load(SLIDE_SEQ_LUNG_ST_MODEL_PATH, st_adata)


st_adata.obsm["proportions"] = st_model.get_proportions(keep_additional=True)
ct_list = st_adata.obsm["proportions"].columns
for ct in ct_list:
    data = st_adata.obsm["proportions"][ct].values
    st_adata.obs[ct] = data

st_adata.obsm["fine_proportions"] = st_model.get_fine_celltypes(sc_model)
ct_list = st_adata.obsm["fine_proportions"].columns
for ct in ct_list:
    data = st_adata.obsm["fine_proportions"][ct].values
    st_adata.obs[ct] = data


# st_adata.var['betas'] = st_model.module.beta.detach().cpu().numpy()


# for ct, g in st_model.get_gamma().items():
#     st_adata.obsm[f"{ct}_gamma"] = g


print('Saving AnnData...')
st_adata.write(SLIDE_SEQ_LUNG_DESTVI_ADATA_PATH)

print('Loading AnnData...')
st_adata = sc.read_h5ad(SLIDE_SEQ_LUNG_DESTVI_ADATA_PATH)

