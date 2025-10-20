import harreman
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
runtime_dict_total['Harreman (CPU)'] = {}
runtime_dict_total['Harreman (GPU)'] = {}

for gene in genes:
    
    print(f'Running Harreman on: {gene} genes...')

    # Imports
    url = "https://github.com/YosefLab/scVI-data/blob/master/rodriques_slideseq.h5ad?raw=true"
    adata = sc.read("rodriques_slideseq.h5ad", backup_url=url)
    adata.obs["total_counts"] = np.asarray(adata.X.sum(1)).ravel()

    adata.layers["csc_counts"] = adata.X.tocsc()

    # renormalize the data for expression viz on plots
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    adata.obsm['spatial'] = np.array(adata.obsm['spatial'])
    
    for device in ['cpu', 'cuda']:
        
        print(device)

        adata_ha = adata.copy()

        # Harreman
        start = time.time()

        harreman.tl.compute_knn_graph(adata_ha, 
                            compute_neighbors_on_key="spatial", 
                            n_neighbors=300,
                            weighted_graph=False)

        harreman.hs.compute_local_autocorrelation(adata_ha, layer_key="csc_counts", model='bernoulli', umi_counts_obs_key="total_counts", device=device)

        hs_results = adata_ha.uns['gene_autocorrelation_results']
        hs_genes = hs_results.loc[hs_results.Z_FDR < 0.05].sort_values('Z', ascending=False).head(gene).index

        harreman.hs.compute_local_correlation(adata_ha, genes=hs_genes, device=device)

        harreman.hs.create_modules(adata_ha, min_gene_threshold=20, core_only=False, fdr_threshold=0.05)

        harreman.hs.calculate_module_scores(adata_ha, device=device)

        runtime = time.time()-start
        key = 'Harreman (CPU)' if device == 'cpu' else 'Harreman (GPU)'
        runtime_dict_total[key][gene] = runtime


with open(os.path.join(DATA_PATH, 'Harreman_runtime_n_genes_dict.pkl'), 'wb') as f:
    pickle.dump(runtime_dict_total, f)

print("Finished successfully.")
