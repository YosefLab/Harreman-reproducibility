import sys
sys.path.insert(0, '/home/projects/nyosef/oier/new_destvi_updated/scvi-tools/src')

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scvi.model import CondSCVI, DestVI
import torch
import destvi_utils
from scipy.stats import ks_2samp, zscore
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import gseapy
from plotnine import *

BASE_PATH = "/home/projects/nyosef/oier/Harreman_files/Slide_seq_lung"
ADATA_PATH = os.path.join(BASE_PATH, 'h5ads')
DATA_PATH = os.path.join(BASE_PATH, 'data')
MODELS_PATH = os.path.join(BASE_PATH, 'models')
PLOTS_PATH = os.path.join(BASE_PATH, 'plots')
SLIDE_SEQ_LUNG_DESTVI_ADATA_PATH = os.path.join(ADATA_PATH, 'Slide_seq_lung_DestVI_v2_adata.h5ad')
SLIDE_SEQ_LUNG_SC_MODEL_PATH = os.path.join(MODELS_PATH, 'Slide_seq_lung_DestVI_v2_sc')
SLIDE_SEQ_LUNG_ST_MODEL_PATH = os.path.join(MODELS_PATH, 'Slide_seq_lung_DestVI_v2_st')

msigdb_data_path = "/home/projects/nyosef/oier/MSigDB_data"

def de_genes(
    st_model,
    mask,
    ct,
    threshold=0.0,
    st_adata=None,
    mask2=None,
    key=None,
    N_sample=10,
    pseudocount=0.01,
    key_proportions="proportions",
):
    """
    Function to compute differential expressed genes from generative model.
    For further reference check [Lopez22]_.

    Parameters
    ----------
    st_adata
        Spatial sequencing dataset with proportions in obsm[key_proportions]. If not provided uses data in st_model.
    st_model
        Trained destVI model
    mask
        Mask for subsetting the spots to condition 1 in differential expression.
    mask2
        Mask for subsetting the spots to condition 2 in differential expression (reference). If none, inverse of mask.
    ct
        Cell type for which differential expression is computed
    threshold
        Proportion threshold to subset to spots with this amount of cell type proportion
    key
        Key to store values in st_adata.uns[key]. If None returns pandas dataframe with DE results. Defaults to None
    N_sample
        N_samples drawn from generative model to simulate expression values.
    pseudocount
        Pseudocount added at computation of logFC. Increasing leads to lower logFC of lowly expressed genes.
    key_proportions
        Obsm key pointing to cell-type proportions.

    Returns
    -------
    res
        If key is None. Pandas dataframe containing results of differential expression.
        Dataframe columns are "log2FC", "pval", "score".
        If key is provided. mask, mask2 and de_results are stored in st_adata.uns[key]. Dictionary keys are
        "mask_active", "mask_rest", "de_results".

    """

    # get statistics
    if mask2 is None:
        mask2 = ~mask

    if st_adata is None:
        st_adata = st_model.adata
        st_adata.obsm[key_proportions] = st_model.get_proportions()
    else:
        if key_proportions not in st_adata.obsm:
            raise ValueError(
                f"Please provide cell type proportions in st_adata.obsm[{key_proportions}] and rerun."
            )

    if st_model.registry_["setup_args"]["layer"]:
        expression = st_adata.layers[st_model.registry_["setup_args"]["layer"]]
    else:
        expression = st_adata.X

    mask = np.logical_and(mask, st_adata.obsm[key_proportions][ct] > threshold)
    mask2 = np.logical_and(mask2, st_adata.obsm[key_proportions][ct] > threshold)

    avg_library_size = np.mean(np.sum(expression, axis=1).flatten())
    exp_px_r = st_model.module.px_r.detach().exp().cpu().numpy()
    imputations = st_model.get_scale_for_ct(ct).values
    mean = avg_library_size * imputations

    concentration = torch.tensor(avg_library_size * imputations / exp_px_r)
    rate = torch.tensor(1.0 / exp_px_r)

    # slice conditions
    N_mask = N_unmask = N_sample

    def simulation(mask_, N_mask_):
        # generate
        simulated = (
            torch.distributions.Gamma(concentration=concentration[mask_], rate=rate)
            .sample((N_mask_,))
            .cpu()
            .numpy()
        )
        simulated = np.log(simulated + 1)
        simulated = simulated.reshape((-1, simulated.shape[-1]))
        return simulated

    simulated_case = simulation(mask, N_mask)
    simulated_control = simulation(mask2, N_unmask)

    de = np.array(
        [
            ks_2samp(
                simulated_case[:, gene],
                simulated_control[:, gene],
                alternative="two-sided",
                mode="asymp",
            )
            for gene in range(simulated_control.shape[1])
        ]
    )
    lfc = np.log2(pseudocount + mean[mask].mean(0)) - np.log2(
        pseudocount + mean[mask2].mean(0)
    )
    res = pd.DataFrame(
        data=np.vstack([lfc, de[:, 0], de[:, 1]]),
        columns=st_adata.var.index,
        index=["log2FC", "score", "pval"],
    ).T

    # Store results in st_adata
    if key is not None:
        st_adata.uns[key] = {}
        st_adata.uns[key]["de_results"] = res.sort_values(by="score", ascending=False)
        st_adata.uns[key]["mask_active"] = mask
        st_adata.uns[key]["mask_rest"] = mask2
        return st_adata
    else:
        return res


deconv_adata = sc.read_h5ad(os.path.join(ADATA_PATH, 'Slide_seq_lung_ct_Harreman_adata.h5ad'))
st_adata = sc.read_h5ad(SLIDE_SEQ_LUNG_DESTVI_ADATA_PATH)
st_model = DestVI.load(SLIDE_SEQ_LUNG_ST_MODEL_PATH, st_adata)

def get_ct_proportion_thershold(proportion_df):
    
    mean = proportion_df.mean()
    std = proportion_df.std()
    
    ct_thresholds = {col: (mean[col] + std[col]) for col in proportion_df.columns}
    
    return ct_thresholds

ct_thresholds = get_ct_proportion_thershold(st_adata.obsm['proportions'])
ct_interacting_cell_results_np_m_cs = pd.read_csv(os.path.join(DATA_PATH, 'Slide_seq_lung_ct_Harreman_ct_interacting_cell_results_np_m_cs.csv'), index_col=0)
cell_com_m_df = deconv_adata.uns['ct_ccc_results']['cell_com_df_m_sig'].copy()
ct_pairs_m_set = set([tuple(x) for x in cell_com_m_df[['Cell Type 1', 'Cell Type 2']].values])

metabolites = ['L-Arginine', 'L-Lactate', 'Sodium_calcium exchange']
metabolite = metabolites[1]

ct_pairs_m_set = set([tuple(x) for x in cell_com_m_df[cell_com_m_df['metabolite'] == metabolite][['Cell Type 1', 'Cell Type 2']].values])
cell_types = list(set(ct for cell_type_pair in ct_pairs_m_set for ct in cell_type_pair))

de_results_dict = {}
for ct in cell_types:
    print(ct)
    rows = [row for row in ct_interacting_cell_results_np_m_cs.index if ct in row]
    st_ct_adata = st_adata[deconv_adata.obs.loc[rows].barcodes.tolist()].copy()
    cols = [col for col in ct_interacting_cell_results_np_m_cs.columns if (metabolite in col) and (ct in col)]
    selected_spots = ct_interacting_cell_results_np_m_cs.loc[rows, cols][ct_interacting_cell_results_np_m_cs.loc[rows, cols].sum(axis=1) != 0].index
    rest = ct_interacting_cell_results_np_m_cs.loc[rows, cols][ct_interacting_cell_results_np_m_cs.loc[rows, cols].sum(axis=1) == 0].index
    mask = st_adata.obs_names.isin(deconv_adata.obs.loc[selected_spots].barcodes.tolist())
    mask2 = st_adata.obs_names.isin(deconv_adata.obs.loc[rest].barcodes.tolist())

    if '_' in ct:
        ct = ct.replace('_', '/')
        
    if (np.logical_and(mask, st_adata.obsm['proportions'][ct] > ct_thresholds[ct]).sum() == 0) or (np.logical_and(mask2, st_adata.obsm['proportions'][ct] > ct_thresholds[ct]).sum() == 0):
        continue

    _ = de_genes(
        st_model, mask=mask, mask2=mask2, threshold=ct_thresholds[ct], ct=ct, key=ct
    )

    res = st_adata.uns[ct]["de_results"]
    res['adj_pval'] = multipletests(res["pval"], method="fdr_bh")[1]

    de_results_dict[ct] = res


writer=pd.ExcelWriter(os.path.join(DATA_PATH, f"{metabolite.replace('-', '_').replace(' ', '_')}_Slide_seq_lung_ct_Harreman_DestVI_DE_results.xlsx"))
for key, df in de_results_dict.items():
    df.to_excel(writer,sheet_name=key.replace('/', '_'))
writer.close()

print('Finished successfully.')
