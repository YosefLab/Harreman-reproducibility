import harreman
import os
import numpy as np
import pandas as pd
import scanpy as sc
from statsmodels.stats.multitest import multipletests

BASE_PATH = "/home/projects/nyosef/oier/Harreman_files/Slide_seq_lung"
SC_REF_PATH = os.path.join(BASE_PATH, 'sc_reference')
ADATA_PATH = os.path.join(BASE_PATH, 'h5ads')
DATA_PATH = os.path.join(BASE_PATH, 'data')

from_dataset_to_filename = {
    'all': 'Slide_seq_lung_sc_ref_scVI_adata.h5ad',
    'only_lung': 'Slide_seq_lung_sc_ref_revision_scVI_adata.h5ad',
}

file = pd.read_excel(os.path.join(DATA_PATH, "Sodium_calcium_exchange_Slide_seq_lung_ct_Harreman_no_deconv_DestVI_revision_DE_results.xlsx"), sheet_name=None, index_col=0)
df = file['TAM']

corr_p_vals = multipletests(df["pval"], method="fdr_bh")
min_score = np.min(df["score"][corr_p_vals[0]])

signatures = {
    'Sodium/calcium exchange': df[(df['score'] > min_score) & (df['log2FC'] > 0)].sort_values('log2FC', ascending=False)[:100].index.tolist(),
}

for dataset, filename in from_dataset_to_filename.items():

    adata = sc.read_h5ad(os.path.join(ADATA_PATH, filename))
    mac_adata = adata[adata.obs['cell_type_coarse'] == 'TAM'].copy()

    norm = sc.pp.normalize_total(mac_adata, target_sum=1e4, inplace=False)
    mac_adata.layers["normalized"] = norm["X"]
    mac_adata.layers["log_norm"] = sc.pp.log1p(norm["X"], copy=True)

    harreman.vs.signatures_from_file(mac_adata, dicts=[signatures])

    harreman.vs.analyze_vision(
        adata=mac_adata,
        norm_data_key="log_norm",
        signature_varm_key="signatures",
        scores_only=True,
    )

    for sig in signatures.keys():
        mac_adata.obs[sig] = mac_adata.obsm['vision_signatures'][sig]

    mac_adata.write(os.path.join(ADATA_PATH, f'Slide_seq_lung_sc_ref_revision_{dataset}_macrophage_visionpy_adata.h5ad'))
