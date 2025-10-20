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


tutorials = ['Spatial', 'Spatial_subset_1', 'Spatial_subset_2', 'CD4']

runtime_dict_total = {}
runtime_dict_total['Harreman (CPU)'] = {}
runtime_dict_total['Harreman (GPU)'] = {}

for tutorial in tutorials:
    
    print(f'Running Harreman on: {tutorial}...')

    if (tutorial == 'Spatial') or ('subset' in tutorial):

        # Imports
        url = "https://github.com/YosefLab/scVI-data/blob/master/rodriques_slideseq.h5ad?raw=true"
        adata = sc.read("rodriques_slideseq.h5ad", backup_url=url)
        adata.obs["total_counts"] = np.asarray(adata.X.sum(1)).ravel()

        adata.layers["csc_counts"] = adata.X.tocsc()

        # renormalize the data for expression viz on plots
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

        adata.obsm['spatial'] = np.array(adata.obsm['spatial'])

        min_x = min(adata.obsm['spatial'][:,0])
        max_x = max(adata.obsm['spatial'][:,0])

        min_y = min(adata.obsm['spatial'][:,1])
        max_y = max(adata.obsm['spatial'][:,1])

        if tutorial == 'Spatial_subset_1':
            adata = adata[(adata.obsm['spatial'][:,0] > (max_x - min_x)/2) & (adata.obsm['spatial'][:,1] > 5*(max_y + min_y)/7),].copy()
        elif tutorial == 'Spatial_subset_2':
            adata = adata[(adata.obsm['spatial'][:,0] > 3*(max_x - min_x)/4) & (adata.obsm['spatial'][:,1] > 2*(max_y + min_y)/3),].copy()

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
            hs_genes = hs_results.loc[hs_results.Z_FDR < 0.05].sort_values('Z', ascending=False).head(500).index

            harreman.hs.compute_local_correlation(adata_ha, genes=hs_genes, device=device)

            harreman.hs.create_modules(adata_ha, min_gene_threshold=20, core_only=False, fdr_threshold=0.05)

            harreman.hs.calculate_module_scores(adata_ha, device=device)

            runtime = time.time()-start
            key = 'Harreman (CPU)' if device == 'cpu' else 'Harreman (GPU)'
            runtime_dict_total[key][tutorial] = runtime

    elif tutorial == 'CD4':

        # Imports
        mdata = mu.read_10x_h5(os.path.join(HOTSPOT_TUTO_DATA, "5k_pbmc_protein_v3_nextgem_filtered_feature_bc_matrix.h5"))
        mdata.var_names_make_unique()

        adata = mdata.mod["rna"]
        adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

        from muon import prot as pt
        pt.pp.clr(mdata['prot'])
        prot_data = mdata.mod["prot"]

        is_cd4 = np.asarray(
            (prot_data[:, 'CD14_TotalSeqB'].X.A < 2) &
            (prot_data[:, 'CD4_TotalSeqB'].X.A > 1) &
            (prot_data[:, 'CD3_TotalSeqB'].X.A > 1)
        ).ravel()

        adata_cd4 = adata[is_cd4]
        sc.pp.filter_cells(adata_cd4, min_genes=1000)
        sc.pp.filter_genes(adata_cd4, min_cells=10)
        adata_cd4 = adata_cd4[adata_cd4.obs.pct_counts_mt < 16].copy()

        adata_cd4.layers["counts"] = adata_cd4.X.copy()
        sc.pp.normalize_total(adata_cd4)
        sc.pp.log1p(adata_cd4)
        adata_cd4.layers["log_normalized"] = adata_cd4.X.copy()
        sc.pp.scale(adata_cd4)
        sc.tl.pca(adata_cd4)

        sc.tl.pca(adata_cd4, n_comps=10)

        adata_cd4.layers["counts_csc"] = adata_cd4.layers["counts"].tocsc()
        
        for device in ['cpu', 'cuda']:
            
            print(device)

            adata_cd4_ha = adata_cd4.copy()

            # Harreman
            start = time.time()

            harreman.tl.compute_knn_graph(adata_cd4_ha, 
                                compute_neighbors_on_key="X_pca", 
                                n_neighbors=30,
                                weighted_graph=False)
            
            harreman.hs.compute_local_autocorrelation(adata_cd4_ha, layer_key="counts_csc", model='danb', umi_counts_obs_key="total_counts", device=device)

            hs_results = adata_cd4_ha.uns['gene_autocorrelation_results']
            hs_genes = hs_results.loc[hs_results.Z_FDR < 0.05].sort_values('Z', ascending=False).head(500).index

            harreman.hs.compute_local_correlation(adata_cd4_ha, genes=hs_genes, device=device)

            harreman.hs.create_modules(adata_cd4_ha, min_gene_threshold=15, core_only=True, fdr_threshold=0.05)

            harreman.hs.calculate_module_scores(adata_cd4_ha, device=device)

            runtime = time.time()-start
            key = 'Harreman (CPU)' if device == 'cpu' else 'Harreman (GPU)'
            runtime_dict_total[key][tutorial] = runtime


with open(os.path.join(DATA_PATH, 'Harreman_runtime_dict.pkl'), 'wb') as f:
    pickle.dump(runtime_dict_total, f)

print("Finished successfully.")
