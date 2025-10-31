import harreman
import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from scipy.stats import pearsonr, wilcoxon, mannwhitneyu, ranksums, zscore
import random
from sklearn import linear_model
from scipy.stats import hypergeom, zscore
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from plotnine import *
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import fcluster
import math
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


BASE_PATH = "/home/projects/nyosef/oier/Harreman_files/Slide_seq_lung"
ADATA_PATH = os.path.join(BASE_PATH, 'h5ads')
DATA_PATH = os.path.join(BASE_PATH, 'data')
PLOTS_PATH = os.path.join(BASE_PATH, 'plots')


# Imports
adata = harreman.read_h5ad(os.path.join(ADATA_PATH, 'Slide_seq_lung_metab_Hotspot_adata.h5ad'))

# Harreman
harreman.pp.extract_interaction_db(adata, species='human', database='both', verbose=True)

harreman.tl.compute_knn_graph(adata, 
                        compute_neighbors_on_key="spatial", 
                        neighborhood_radius=100,
                        weighted_graph=False,
                        sample_key='sample',
                        verbose=True
                        )

harreman.tl.apply_gene_filtering(adata, layer_key='counts', model='bernoulli', autocorrelation_filt = True, verbose=True)

harreman.tl.compute_gene_pairs(adata, ct_specific = False, verbose=True)

harreman.tl.compute_cell_communication(adata, model='bernoulli', M = 1000, test = "both", layer_key_p_test='counts', layer_key_np_test='log_norm', verbose=True)

harreman.tl.select_significant_interactions(adata, test = "non-parametric", threshold = 0.05)


harreman.tl.compute_interacting_cell_scores(adata, test = "both", compute_significance='parametric', verbose=True)

harreman.tl.compute_interaction_module_correlation(adata, cor_method='pearson', interaction_type='metabolite', test='non-parametric')


harreman.write_h5ad(adata, filename = os.path.join(ADATA_PATH, 'Slide_seq_lung_Harreman_adata.h5ad'))


print("Finished successfully.")
