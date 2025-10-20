import harreman
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl

BASE_PATH = "/home/projects/nyosef/oier/Harreman_files/Slide_seq_lung"
ADATA_PATH = os.path.join(BASE_PATH, 'h5ads')
DATA_PATH = os.path.join(BASE_PATH, 'data')


adata = sc.read_h5ad(os.path.join(ADATA_PATH, 'Slide_seq_lung_metab_Hotspot_adata.h5ad'))
adata = adata[adata.obs['top_module'].isin(['Module 1', 'Module 2'])].copy()

sig_dict = {
    'Kidney': ["CA9", "VIM", "PAX8", "NDUFA4L2", "EGLN3", "ENO2", "SLC16A3", "SLC2A1", "SLC8A1", "AQP1", "CP", "FABP7", "GGT1", "SLC17A3", "PLIN2", "GPX3", "NNMT", "PCSK1N"],
    'Lung': ["NKX2-1", "SFTPB", "SFTPC", "NAPSA", "CLDN18", "KRT5", "TP63", "SOX2", "FGFR1", "MUC1", "TTF1", "CEACAM5", "EPCAM", "KRT7", "SCGB1A1", "CDH1", "FOXA2", "GATA6", "B3GNT7"],
}


def visionpy_pipeline(adata, sig_dict, DATA_PATH):

    if "log_norm" not in adata.layers:
        adata.X = adata.layers['counts']

        norm = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)
        adata.layers["normalized"] = norm["X"]
        adata.layers["log_norm"] = sc.pp.log1p(norm["X"])

    harreman.vs.signatures_from_file(adata, dicts=[sig_dict])

    harreman.vs.analyze_vision(
        adata=adata,
        norm_data_key="log_norm",
        signature_varm_key="signatures",
        scores_only=True,
    )

    adata.write(DATA_PATH)


OUTPUT_PATH = os.path.join(ADATA_PATH, 'kidney_lung_visionpy_adata.h5ad')

visionpy_pipeline(adata, sig_dict, OUTPUT_PATH)


print('Successfully finished.')
