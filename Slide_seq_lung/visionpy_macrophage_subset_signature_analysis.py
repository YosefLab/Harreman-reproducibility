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

SIGNATURES_PATH = "/home/projects/nyosef/oier/signatures"


adata = sc.read_h5ad(os.path.join(ADATA_PATH, 'Slide_seq_lung_metab_Hotspot_adata.h5ad'))

df = pd.read_excel(os.path.join(SIGNATURES_PATH, "Bi_et_al_Table_S3.xlsx"), sheet_name='A', index_col=0)
df.columns = df.iloc[0]
df = df.iloc[1:].copy()

signatures = {}
macrophage_clusters = [clust for clust in df['cluster'].unique() if 'TAM' in clust]
for clust in macrophage_clusters:
    mac_df = df[df['cluster'] == clust].copy()
    mask = (mac_df[['pct exp in cluster', 'pct exp in else']] >= 0.05).sum(1) != 0
    mac_df = mac_df.loc[mask].copy()
    sig_genes = mac_df[(mac_df['log2FC'] > 0) & (mac_df['FDR-adj. p value'] < 0.05)].sort_values('log2FC', ascending=False)[:100].index.tolist()
    signatures[clust] = sig_genes

harreman.vs.signatures_from_file(adata, dicts=[signatures])

harreman.vs.analyze_vision(
    adata=adata,
    norm_data_key="log_norm",
    signature_varm_key="signatures",
    scores_only=True,
)

for sig in signatures.keys():
    adata.obs[sig] = adata.obsm['vision_signatures'][sig]

adata.write(os.path.join(ADATA_PATH, 'Slide_seq_lung_revision_macrophage_subset_visionpy_adata.h5ad'))
