import hotspot
import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import anndata as ad
import pickle
import warnings
warnings.filterwarnings("ignore")


BASE_PATH = '/home/projects/nyosef/oier/Harreman_files/Hotspot_tutorial_analysis'
DATA_PATH = os.path.join(BASE_PATH, 'data')
ADATA_PATH = os.path.join(BASE_PATH, 'h5ads')

HOTSPOT_TUTO_DATA = '/home/projects/nyosef/oier/Hotspot_tutorial_data'


genes = [200, 400, 600, 800]

runtime_dict_total = {}
runtime_dict_total['Harreman'] = {}
runtime_dict_total['Hotspot'] = {}

for gene in genes:

    # Imports
    url = "https://github.com/YosefLab/scVI-data/blob/master/rodriques_slideseq.h5ad?raw=true"
    adata = sc.read("rodriques_slideseq.h5ad", backup_url=url)
    adata.obs["total_counts"] = np.asarray(adata.X.sum(1)).ravel()

    adata.layers["csc_counts"] = adata.X.tocsc()

    # renormalize the data for expression viz on plots
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    adata.obsm['spatial'] = np.array(adata.obsm['spatial'])

    adata_ho = adata.copy()

    # Hotspot
    start = time.time()

    hs = hotspot.Hotspot(
        adata_ho,
        layer_key="csc_counts",
        model='bernoulli',
        latent_obsm_key="spatial",
        umi_counts_obs_key="total_counts",
    )

    hs.create_knn_graph(
        weighted_graph=False, n_neighbors=300, approx_neighbors=False,
    )

    hs_results = hs.compute_autocorrelations(jobs=1)
    hs_genes = hs_results.loc[hs_results.FDR < 0.05].sort_values('Z', ascending=False).head(gene).index

    lcz = hs.compute_local_correlations(hs_genes, jobs=1)

    modules = hs.create_modules(
        min_gene_threshold=20, core_only=False, fdr_threshold=0.05
    )

    module_scores = hs.calculate_module_scores()

    runtime = time.time()-start
    runtime_dict_total['Hotspot'][gene] = runtime


with open(os.path.join(DATA_PATH, 'Hotspot_runtime_n_genes_dict.pkl'), 'wb') as f:
    pickle.dump(runtime_dict_total, f)

print("Finished successfully.")
