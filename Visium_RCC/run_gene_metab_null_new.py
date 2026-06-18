"""
run_gene_metab_null.py
======================
Standalone script to run the gene-metabolite shuffling null analysis
for the Harreman MSI validation (Visium RCC dataset).

Loads precomputed all_results (spatial permutation test already done),
runs harreman_null_test with n_perm=100 per sample, updates all_results
in place, corrects observed values to match the spatial permutation test,
and saves the updated all_results to disk.

Usage
-----
    python run_gene_metab_null.py

Output
------
    all_results_with_gene_metab_null.pkl   (updated all_results dict)

Dependencies
------------
    harreman, scanpy, anndata, numpy, pandas, scipy, sklearn, liana
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import harreman
import liana as li
from scipy.stats import zscore, spearmanr
from scipy.stats.contingency import chi2_contingency
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_PATH   = "/home/projects/nyosef/oier/Harreman_files/Revision/Visium_RCC"
ADATA_PATH  = os.path.join(BASE_PATH, "h5ads")
DATA_PATH   = os.path.join(BASE_PATH, "data")

# Path for the final output
ALL_RESULTS_OUT = os.path.join(DATA_PATH, "all_results_with_gene_metab_null_new.pkl")

N_PERM      = 100
N_STD       = 1.0
BANDWIDTH   = 500
CUTOFF      = 0.1
RANDOM_STATE = 42

# ── Helper functions (copied from notebook) ────────────────────────────────

def gmm_binarize_matrix(X, k=2, prob_threshold=0.5):
    X_bin = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j].reshape(-1, 1)
        mask = ~np.isnan(col[:, 0])
        col_valid = col[mask]
        if len(col_valid) < 10:
            continue
        gmm = GaussianMixture(n_components=k, random_state=0)
        gmm.fit(col_valid)
        probs = gmm.predict_proba(col_valid)
        high_comp = np.argmax(gmm.means_.flatten())
        bin_vals = (probs[:, high_comp] > prob_threshold).astype(int)
        X_bin[mask, j] = bin_vals
    return X_bin


def std_binarize_matrix(X, n_std=1.0):
    means = np.nanmean(X, axis=0)
    stds  = np.nanstd(X,  axis=0)
    return (X >= means + n_std * stds).astype(float)


def binarize_and_report(X, var_names, process="std", n_std=1.0, verbose=True):
    if process == "gmm":
        X_bin = gmm_binarize_matrix(X, k=2, prob_threshold=0.5)
    else:
        X_bin = std_binarize_matrix(X, n_std=n_std)
    active_frac = X_bin.mean(axis=0)
    report = pd.DataFrame({"feature": var_names, "active_frac": active_frac})
    if verbose:
        print(f"[binarization] process={process}, n_std={n_std}")
        print(f"  median active-spot fraction : {np.median(active_frac):.3f}")
        print(f"  features with <1% active    : {(active_frac < 0.01).sum()}")
        print(f"  features with >50% active   : {(active_frac > 0.50).sum()}")
    return X_bin, report


def threshold_sensitivity_check_per_sample(adata, samples, sample_col="sample",
                                            n_std_values=(0.5, 1.0, 1.5, 2.0),
                                            layer="raw"):
    summary_rows = []
    for sample in samples:
        s_adata = adata[adata.obs[sample_col] == sample]
        X = np.asarray(s_adata.layers[layer])
        for n in n_std_values:
            X_bin = std_binarize_matrix(X, n_std=n)
            af    = X_bin.mean(axis=0)
            summary_rows.append({
                "sample":             sample,
                "n_std":              n,
                "median_active_frac": np.median(af),
                "mean_active_frac":   np.mean(af),
                "pct_below_1pct":     (af < 0.01).mean() * 100,
                "pct_above_50pct":    (af > 0.50).mean() * 100,
            })
    return pd.DataFrame(summary_rows)


def classify_samples_by_sparsity(sensitivity_summary, n_std=1.0,
                                  sparse_threshold=0.03):
    df = (sensitivity_summary
          .query("n_std == @n_std")
          [["sample", "median_active_frac"]]
          .copy())
    df["recommended_process"] = np.where(
        df["median_active_frac"] < sparse_threshold, "gmm", "std")
    return df


def harreman_scores_from_adata(adata, process="std", n_std=1.0):
    scores = adata.uns["interacting_cell_results"]["np"]["m"]["cs"].copy()
    X_bin, report = binarize_and_report(scores, adata.uns["metabolites"],
                                        process=process, n_std=n_std)
    return pd.DataFrame(X_bin, index=adata.obs_names,
                        columns=adata.uns["metabolites"]), report


def compute_chi2_stat(df):
    contingency = pd.crosstab(df["msi_module"], df["harreman_group"])
    chi2, _, _, _ = chi2_contingency(contingency)
    return chi2


def compute_soft_overlap(msi_scores, harreman_scores):
    common_idx = msi_scores.index.intersection(harreman_scores.index)
    A = msi_scores.loc[common_idx]
    B = harreman_scores.loc[common_idx]
    r_matrix = pd.DataFrame(index=A.columns, columns=B.columns, dtype=float)
    for col_a in A.columns:
        for col_b in B.columns:
            r, _ = spearmanr(A[col_a], B[col_b])
            r_matrix.loc[col_a, col_b] = r
    mean_max_r = r_matrix.max(axis=1).mean()
    return mean_max_r, r_matrix


def compute_ari_nmi(df):
    ari = adjusted_rand_score(df["msi_module"], df["harreman_group"])
    nmi = normalized_mutual_info_score(df["msi_module"], df["harreman_group"])
    return ari, nmi


def compute_all_metrics(df, msi_scores, harreman_scores):
    chi2 = compute_chi2_stat(df)
    ari, nmi = compute_ari_nmi(df)
    mean_max_r, r_matrix = compute_soft_overlap(msi_scores, harreman_scores)
    return {"chi2": chi2, "ari": ari, "nmi": nmi,
            "mean_max_r": mean_max_r, "r_matrix": r_matrix}


def shuffle_gene_metabolite_matrix_kmeans(gene_metab_df, gene_stats_df,
                                          n_bins=20, rng=None,
                                          random_state=0):
    if rng is None:
        rng = np.random.default_rng(0)
    genes = gene_metab_df.index.intersection(gene_stats_df.index)
    gene_metab_df = gene_metab_df.loc[genes]
    features = gene_stats_df.loc[genes].copy().dropna()
    gene_metab_df = gene_metab_df.loc[features.index]
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    kmeans = KMeans(n_clusters=n_bins, random_state=random_state, n_init="auto")
    bins = pd.Series(kmeans.fit_predict(X), index=features.index)
    shuffled = gene_metab_df.copy()
    for metab in shuffled.columns:
        col = shuffled[metab].copy()
        for b in bins.unique():
            idx = bins[bins == b].index
            if len(idx) > 1:
                values = col.loc[idx].values.copy()
                rng.shuffle(values)
                col.loc[idx] = values
        shuffled[metab] = col
    return shuffled


HARREMAN_RESULT_KEYS = [
    "interacting_cell_results",
    "ccc_results",
    "gene_pairs", "gene_pairs_ind",
    "gene_pairs_per_metabolite",
    "gene_pairs_sig", "gene_pairs_sig_ind", "gene_pairs_sig_names",
    "gene_pair_dict",
    "cell_type_pairs", "gene_pairs_per_ct_pair",
    "lcs", "lc_zs", "lc_z_pvals", "lc_z_FDR",
    "gene_modules", "gene_modules_dict",
    "linkage", "mod_reordered", "modules",
    "interaction_module_correlation_FDR",
    "interaction_module_correlation_coefs",
    "interaction_module_correlation_pvals",
    "metabolites",
]


def _set_database_on_adata(adata, shuffled_db):
    adata.varm["database"] = shuffled_db.copy()
    for key in HARREMAN_RESULT_KEYS:
        if key in adata.uns:
            del adata.uns[key]
    if "import_export" in adata.uns:
        ie_metabolites = [col for col in shuffled_db.columns
                          if (shuffled_db[col] == 2.0).any()]
        if ie_metabolites:
            ie_genes_per_metab = {
                col: shuffled_db.index[shuffled_db[col] == 2.0].tolist()
                for col in ie_metabolites
            }
            max_len = max(len(v) for v in ie_genes_per_metab.values())
            ie_df = pd.DataFrame({
                f"IMPORT_EXPORT{i}": pd.Series([
                    ie_genes_per_metab[col][i]
                    if i < len(ie_genes_per_metab[col]) else None
                    for col in ie_metabolites
                ])
                for i in range(max_len)
            })
            adata.uns["import_export"] = ie_df
        else:
            adata.uns["import_export"] = pd.DataFrame()
    if "ligand" in adata.uns and "LR_database" in adata.uns:
        lr_cols = [c for c in shuffled_db.columns
                   if c in set(adata.uns["LR_database"].index)]
        if lr_cols:
            lr_sub        = shuffled_db[lr_cols]
            ligand_dict   = {col: lr_sub.index[lr_sub[col] == 1.0].tolist()
                             for col in lr_cols}
            receptor_dict = {col: lr_sub.index[lr_sub[col] == -1.0].tolist()
                             for col in lr_cols}
            max_len_l = max((len(v) for v in ligand_dict.values()), default=1)
            max_len_r = max((len(v) for v in receptor_dict.values()), default=1)
            adata.uns["ligand"] = pd.DataFrame(
                {col: pd.Series(ligand_dict[col]) for col in lr_cols}
            ).T.reindex(columns=range(max_len_l))
            adata.uns["receptor"] = pd.DataFrame(
                {col: pd.Series(receptor_dict[col]) for col in lr_cols}
            ).T.reindex(columns=range(max_len_r))


def run_harreman(adata, shuffled, process="std", n_std=1.0):
    harreman.tl.compute_knn_graph(adata,
                                  compute_neighbors_on_key="spatial",
                                  n_neighbors=5,
                                  weighted_graph=False,
                                  sample_key="sample")
    harreman.tl.apply_gene_filtering(adata, layer_key="counts",
                                     model="danb",
                                     autocorrelation_filt=True)
    harreman.tl.compute_gene_pairs(adata, ct_specific=False)
    harreman.tl.compute_cell_communication(
        adata, model="danb", M=1000, test="both",
        layer_key_p_test="counts",
        layer_key_np_test="log_norm")
    harreman.tl.select_significant_interactions(
        adata, test="non-parametric", threshold=0.05)
    harreman.tl.compute_interacting_cell_scores(
        adata, test="both", compute_significance="parametric",
        verbose=False)

    scores_df, _ = harreman_scores_from_adata(adata, process=process,
                                              n_std=n_std)
    metab_scores_adata = ad.AnnData(scores_df)
    metab_scores_adata.obs["sample"]   = adata.obs["sample"]
    metab_scores_adata.obsm["spatial"] = adata.obsm["spatial"]

    harreman.tl.compute_knn_graph(metab_scores_adata,
                                  compute_neighbors_on_key="spatial",
                                  n_neighbors=5,
                                  weighted_graph=False,
                                  sample_key="sample")
    harreman.hs.compute_local_autocorrelation(metab_scores_adata,
                                              model="bernoulli")
    harreman.hs.compute_local_correlation(
        metab_scores_adata, genes=metab_scores_adata.var_names)
    harreman.hs.create_modules(metab_scores_adata, min_gene_threshold=20)
    harreman.hs.calculate_module_scores(metab_scores_adata)

    if metab_scores_adata.obsm["module_scores"].empty:
        return None, None

    module_scores = metab_scores_adata.obsm["module_scores"]
    top_group = (
        pd.DataFrame(zscore(module_scores, axis=0),
                     index=module_scores.index,
                     columns=module_scores.columns)
        .idxmax(axis=1)
    )
    return top_group, module_scores


def harreman_null_test(adata, df_template, msi_scores,
                       observed_metrics,
                       n_perm=100, process="std", n_std=1.0,
                       random_state=0):
    """
    Gene-metabolite shuffling null.

    Uses precomputed observed_metrics (from the spatial permutation test)
    to ensure consistency between panels C and I. Does NOT rerun the
    pipeline on the real database.
    """
    rng = np.random.default_rng(random_state)

    harreman.pp.extract_interaction_db(adata, species="human", database="both")
    gene_metab_df = adata.varm["database"].copy()

    # Build gene stats for kmeans-stratified shuffling
    X = adata.layers["log_norm"]
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    gene_stats_df = pd.DataFrame({
        "mean": X.mean(axis=0),
        "pct":  (X > 0).mean(axis=0),
    }, index=adata.var_names)

    # Use precomputed observed metrics — do NOT rerun on real database
    obs_metrics = observed_metrics

    null      = {"chi2": [], "ari": [], "nmi": [], "mean_max_r": []}
    n_skipped = 0

    for i in range(n_perm):
        print(f"  Permutation {i+1}/{n_perm}", flush=True)

        shuffled = shuffle_gene_metabolite_matrix_kmeans(
            gene_metab_df,
            gene_stats_df=gene_stats_df,
            n_bins=20,
            rng=rng,
        )

        adata_perm = adata.copy()
        _set_database_on_adata(adata_perm, shuffled)

        perm_labels, perm_scores = run_harreman(
            adata_perm, shuffled, process=process, n_std=n_std)

        if perm_labels is None:
            n_skipped += 1
            for k in ["chi2", "ari", "nmi", "mean_max_r"]:
                null[k].append(0.0)
            continue

        common_perm = df_template.index.intersection(perm_labels.index)
        df_perm     = df_template.loc[common_perm].copy()
        df_perm["harreman_group"] = perm_labels.loc[common_perm].values
        msi_scores_perm = msi_scores.loc[common_perm]

        m = compute_all_metrics(df_perm, msi_scores_perm, perm_scores)
        for k in ["chi2", "ari", "nmi", "mean_max_r"]:
            null[k].append(m[k])

    null = {k: np.array(v) for k, v in null.items()}
    if n_skipped > 0:
        print(f"  [null] {n_skipped}/{n_perm} permutations produced no modules "
              f"— assigned concordance = 0.0")
    p_empirical = {
        k: np.mean(null[k] >= obs_metrics[k])
        for k in ["chi2", "ari", "nmi", "mean_max_r"]
    }
    return {"observed": obs_metrics, "null": null, "p_empirical": p_empirical}


# ── Main ───────────────────────────────────────────────────────────────────

def feature_filtering(adata, sample_col="sample", min_frac=0.05,
                      var_quantile=0.1):
    metabs_to_keep = np.zeros(adata.shape[1], dtype=bool)
    for sample in adata.obs[sample_col].unique():
        adata_s = adata[adata.obs[sample_col] == sample]
        X = np.asarray(adata_s.X)
        thresh = np.percentile(X, 1)
        expressed = (X > thresh).sum(axis=0)
        min_cells = int(min_frac * X.shape[0])
        metabs_to_keep |= expressed >= min_cells
    adata = adata[:, metabs_to_keep].copy()
    var = np.var(np.asarray(adata.X), axis=0)
    keep_var = var > np.quantile(var, var_quantile)
    adata = adata[:, keep_var].copy()
    return adata


def msi_scores_from_adata(adata, process="std", n_std=1.0):
    scores = np.asarray(adata.layers["raw"])
    X_bin, report = binarize_and_report(scores, adata.var_names,
                                        process=process, n_std=n_std)
    return pd.DataFrame(X_bin, index=adata.obs_names,
                        columns=adata.var_names), report


def harreman_sample_scores_from_adata(adata_, obs_names, process="std",
                                      n_std=1.0):
    scores = adata_.uns["interacting_cell_results"]["np"]["m"]["cs"].copy()
    idx = [list(obs_names).index(name) for name in adata_.obs_names]
    scores = scores[idx]
    X_bin, report = binarize_and_report(scores, adata_.uns["metabolites"],
                                        process=process, n_std=n_std)
    return pd.DataFrame(X_bin, index=adata_.obs_names,
                        columns=adata_.uns["metabolites"]), report


def merge_modules_by_correlation(adata, corr_threshold=0.8,
                                 score_key="module_scores"):
    scores = adata.obsm[score_key]
    module_names = list(scores.columns)
    n = len(module_names)
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(scores.iloc[:, i], scores.iloc[:, j])
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r
    visited = [False] * n
    groups  = []
    for i in range(n):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and corr_matrix[i, j] >= corr_threshold:
                group.append(j)
                visited[j] = True
        groups.append(group)
    super_module_dict = {
        sm_id + 1: [int(module_names[idx].split(' ')[1]) for idx in group]
        for sm_id, group in enumerate(groups)
    }
    print(f"[merge_modules] {n} modules → {len(super_module_dict)} "
          f"super-modules (threshold r={corr_threshold})")
    for sm_id, members in super_module_dict.items():
        print(f"  SM{sm_id}: {members}")
    harreman.hs.calculate_super_module_scores(
        adata, super_module_dict=super_module_dict)
    super_scores = adata.obsm["super_module_scores"]
    adata.uns["module_corr_matrix"] = corr_matrix
    adata.uns["gene_modules_sm"]    = super_module_dict
    return super_module_dict, super_scores, corr_matrix


def run_per_sample_msi_harreman_pipeline(adata_, corr_threshold=0.8):
    harreman.tl.compute_knn_graph(adata_,
                                  compute_neighbors_on_key="spatial",
                                  n_neighbors=5,
                                  weighted_graph=False,
                                  sample_key="sample")
    harreman.hs.compute_local_autocorrelation(adata_, model="bernoulli")
    res = adata_.uns["gene_autocorrelation_results"]
    metabolites = (res.loc[res.Z_FDR < 0.01]
                     .sort_values("Z", ascending=False).index)
    harreman.hs.compute_local_correlation(adata_, genes=metabolites)
    harreman.hs.create_modules(adata_, min_gene_threshold=15)
    harreman.hs.calculate_module_scores(adata_)
    super_module_dict, super_scores, corr_matrix = merge_modules_by_correlation(
        adata_, corr_threshold=corr_threshold)
    adata_.obsm["super_module_scores"] = super_scores
    adata_.uns["gene_modules_sm"]  = super_module_dict
    adata_.uns["module_corr_matrix"] = corr_matrix
    modules = adata_.obsm["super_module_scores"].columns
    adata_.obs[modules] = adata_.obsm["super_module_scores"]
    return adata_


def run_per_sample_harreman_pipeline(adata_):
    harreman.tl.compute_knn_graph(adata_,
                                  compute_neighbors_on_key="spatial",
                                  n_neighbors=5,
                                  weighted_graph=False,
                                  sample_key="sample")
    harreman.hs.compute_local_autocorrelation(adata_, model="bernoulli")
    harreman.hs.compute_local_correlation(adata_, genes=adata_.var_names)
    harreman.hs.create_modules(adata_, min_gene_threshold=15)
    harreman.hs.calculate_module_scores(adata_)
    modules = adata_.obsm["module_scores"].columns
    adata_.obs[modules] = adata_.obsm["module_scores"]
    adata_.obs["top_group"] = (
        pd.DataFrame(zscore(adata_.obsm["module_scores"], axis=0),
                     index=adata_.obsm["module_scores"].index,
                     columns=adata_.obsm["module_scores"].columns)
        .idxmax(axis=1)
    )
    return adata_


def shuffle_labels_within_sample(df, label_col="harreman_group",
                                  sample_col="sample", rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    df_shuffled = df.copy()
    for sample in df[sample_col].unique():
        idx = df[sample_col] == sample
        labels = df.loc[idx, label_col].values.copy()
        rng.shuffle(labels)
        df_shuffled.loc[idx, label_col] = labels
    return df_shuffled


def permute_scores_within_sample(scores_df, sample_labels, rng):
    permuted = scores_df.copy()
    for sample in sample_labels.unique():
        idx = sample_labels[sample_labels == sample].index
        shuffled_idx = rng.permutation(idx)
        permuted.loc[idx] = scores_df.loc[shuffled_idx].values
    return permuted


def permutation_test(df, msi_scores, harreman_scores,
                     n_perm=1000, random_state=0):
    rng = np.random.default_rng(random_state)
    obs = compute_all_metrics(df, msi_scores, harreman_scores)
    null = {"chi2": [], "ari": [], "nmi": [], "mean_max_r": []}
    for _ in range(n_perm):
        df_perm = shuffle_labels_within_sample(
            df, label_col="harreman_group", sample_col="sample", rng=rng)
        null["chi2"].append(compute_chi2_stat(df_perm))
        ari_p, nmi_p = compute_ari_nmi(df_perm)
        null["ari"].append(ari_p)
        null["nmi"].append(nmi_p)
        harreman_perm = permute_scores_within_sample(
            harreman_scores, df["sample"], rng)
        mmr_p, _ = compute_soft_overlap(msi_scores, harreman_perm)
        null["mean_max_r"].append(mmr_p)
    null = {k: np.array(v) for k, v in null.items()}
    p_empirical = {
        k: np.mean(null[k] >= obs[k])
        for k in ["chi2", "ari", "nmi", "mean_max_r"]
    }
    return {"observed": obs, "null": null, "p_empirical": p_empirical}


def main():
    print("=" * 55)
    print("  Harreman MSI validation — full pipeline")
    print("=" * 55)

    # ── Load data ──────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    adata = harreman.read_h5ad(
        os.path.join(ADATA_PATH, "Visium_RCC_Harreman_unrolled.h5ad"))
    msi_neg = sc.read_h5ad(
        os.path.join(ADATA_PATH, "neg_msi_all.h5ad"))
    rna = sc.read_h5ad(
        os.path.join(ADATA_PATH, "neg_rna_new.h5ad"))
    metab_hs_adata = harreman.read_h5ad(
        os.path.join(ADATA_PATH, "Visium_RCC_Metabolic_Hotspot.h5ad"))

    metabolite_annotation = pd.read_excel(
        os.path.join(DATA_PATH, "spatial_metabolites_annotation.xlsx"),
        index_col=0).dropna(subset="Metabolites")
    metabolite_annotation["mz"] = metabolite_annotation["mz"].round(3)
    msi_neg.var = (
        msi_neg.var.reset_index()
        .merge(metabolite_annotation, on="mz", how="left")
        .set_index("index")
    )
    
    # Samples already processed — results already in all_results_with_gene_metab_null.pkl
    existing_samples = ["R_cor", "Z43_T", "Y7_T", "R114_T", "R51_T", "R29_T"]

    # New samples to process — derived automatically from rna obs
    new_samples = [s for s in rna.obs["sample"].unique().tolist()
                if s not in existing_samples]

    print(f"\nExisting samples (will be skipped): {existing_samples}")
    print(f"New samples to process: {new_samples}")

    # ── Feature filtering ──────────────────────────────────────
    print("\n[2/6] Feature filtering...")
    msi_neg = feature_filtering(msi_neg)

    # ── Per-sample binarization strategy ──────────────────────
    print("\n[3/6] Threshold sensitivity analysis...")
    samples = rna.obs["sample"].unique().tolist()
    sensitivity_summary = threshold_sensitivity_check_per_sample(
        msi_neg, samples, sample_col="sample",
        n_std_values=(0.5, 1.0, 1.5, 2.0), layer="raw")
    sample_classes = classify_samples_by_sparsity(
        sensitivity_summary, n_std=1.0, sparse_threshold=0.03)
    process_per_sample = sample_classes.set_index(
        "sample")["recommended_process"]

    # ── Per-sample binarization and MSI AnnData assembly ──────
    print("\n[4/6] Binarization and MSI zonation...")
    msi_sample_adatas = []
    for sample in new_samples:
        s_msi = msi_neg[msi_neg.obs["sample"] == sample].copy()
        assigned_process = process_per_sample.get(sample, "std")
        print(f"  [{sample}] binarization → process={assigned_process}")
        scores_df, _ = msi_scores_from_adata(
            s_msi, process=assigned_process, n_std=N_STD)
        s_adata = ad.AnnData(scores_df)
        s_adata.obs["sample"]       = s_msi.obs["sample"]
        s_adata.obsm["spatial"]     = s_msi.obsm["spatial"]
        msi_sample_adatas.append(s_adata)

    msi_metab_scores_adata = ad.concat(
        msi_sample_adatas, join="outer", fill_value=0)
    msi_metab_scores_adata.obs_names_make_unique()

    # ── MSI zonation per sample ────────────────────────────────
    msi_adatas = {}
    for sample in new_samples:
        print(f"\n  [MSI zonation] {sample}")
        s_adata = msi_metab_scores_adata[
            msi_metab_scores_adata.obs["sample"] == sample].copy()
        s_adata = run_per_sample_msi_harreman_pipeline(
            s_adata, corr_threshold=0.8)
        msi_adatas[sample] = s_adata

    # ── Interpolate MSI super-modules onto RNA spots ───────────
    rna_adatas = {}
    for sample, sample_adata in msi_adatas.items():
        print(f"\n  [MSI→RNA interpolation] {sample}")
        sample_rna = rna[rna.obs["sample"] == sample].copy()
        reference  = sample_adata.obsm["spatial"]
        li.utils.spatial_neighbors(
            sample_rna, bandwidth=BANDWIDTH, cutoff=CUTOFF,
            spatial_key="spatial", reference=reference,
            set_diag=False, standardize=False)
        W = sample_rna.obsm["spatial_connectivities"]
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        W_norm = W.multiply(1 / row_sums[:, None])
        msi_on_rna = W_norm @ sample_adata.obsm["super_module_scores"]
        sample_rna.obsm["msi_module_scores"] = pd.DataFrame(
            msi_on_rna,
            index=sample_rna.obs_names,
            columns=sample_adata.obsm["super_module_scores"].columns)
        sample_rna.obs["top_super_module"] = (
            pd.DataFrame(
                zscore(sample_rna.obsm["msi_module_scores"], axis=0),
                index=sample_rna.obsm["msi_module_scores"].index,
                columns=sample_rna.obsm["msi_module_scores"].columns)
            .idxmax(axis=1)
        )
        sample_rna.uns["gene_modules_sm"] = sample_adata.uns["gene_modules_sm"]
        rna_adatas[sample] = sample_rna

    # ── Harreman zonation per sample ───────────────────────────
    harreman_adatas = {}
    for sample in rna_adatas:
        print(f"\n  [Harreman zonation] {sample}")
        sample_adata = adata[adata.obs["sample"] == sample].copy()
        raw_scores = adata.uns["interacting_cell_results"]["np"]["m"]["cs"]
        idx = [list(adata.obs_names).index(n)
               for n in sample_adata.obs_names]
        raw_scores_sample = raw_scores[idx]
        tmp = ad.AnnData(raw_scores_sample)
        tmp.obs_names   = sample_adata.obs_names
        tmp.var_names   = adata.uns["metabolites"]
        tmp.obs["sample"] = sample
        tmp.layers["raw"] = raw_scores_sample
        har_sensitivity = threshold_sensitivity_check_per_sample(
            tmp, [sample], sample_col="sample",
            n_std_values=(0.5, 1.0, 1.5, 2.0), layer="raw")
        har_classes = classify_samples_by_sparsity(
            har_sensitivity, n_std=1.0, sparse_threshold=0.03)
        assigned_process = har_classes.set_index(
            "sample")["recommended_process"].get(sample, "std")
        scores_df, _ = harreman_sample_scores_from_adata(
            sample_adata, adata.obs_names,
            process=assigned_process, n_std=N_STD)
        metab_scores_adata = ad.AnnData(scores_df)
        metab_scores_adata.obs["sample"]   = sample_adata.obs["sample"]
        metab_scores_adata.obsm["spatial"] = sample_adata.obsm["spatial"]
        metab_scores_adata = run_per_sample_harreman_pipeline(
            metab_scores_adata)
        harreman_adatas[sample] = metab_scores_adata

    # ── Spatial permutation test ───────────────────────────────
    print("\n[5/6] Spatial permutation test (n=1000)...")
    all_results = {}
    for sample, rna_adata in rna_adatas.items():
        print(f"\n  {'#'*50}\n  Sample: {sample}\n  {'#'*50}")
        harreman_adata = harreman_adatas[sample]
        msi_labels     = rna_adata.obs["top_super_module"]
        harreman_labels = harreman_adata.obs["top_group"]
        sample_labels  = harreman_adata.obs["sample"]
        df = pd.concat(
            [sample_labels[msi_labels.index],
             msi_labels,
             harreman_labels.loc[msi_labels.index]], axis=1)
        df.columns = ["sample", "msi_module", "harreman_group"]
        msi_scores_sample = rna_adata.obsm["msi_module_scores"].loc[df.index]
        har_scores_sample = harreman_adata.obsm["module_scores"].loc[df.index]
        perm_results = permutation_test(
            df, msi_scores_sample, har_scores_sample,
            n_perm=1000, random_state=42)
        print(f"\n  Permutation test results — {sample}")
        for metric in ["chi2", "ari", "nmi", "mean_max_r"]:
            obs = perm_results["observed"][metric]
            p   = perm_results["p_empirical"][metric]
            print(f"    {metric:<12}  obs={obs:.4f}  p={p:.4f}")
        all_results[sample] = {"perm": perm_results}

    # ── Gene-metabolite shuffling null ─────────────────────────
    print(f"\n[6/6] Gene-metabolite shuffling null (n={N_PERM})...")
    for sample, rna_adata in rna_adatas.items():
        print(f"\n  {'#'*50}\n  Sample: {sample}\n  {'#'*50}")
        harreman_adata = harreman_adatas[sample]
        msi_labels     = rna_adata.obs["top_super_module"]
        harreman_labels = harreman_adata.obs["top_group"]
        sample_labels  = harreman_adata.obs["sample"]
        df = pd.concat(
            [sample_labels[msi_labels.index],
             msi_labels,
             harreman_labels.loc[msi_labels.index]], axis=1)
        df.columns = ["sample", "msi_module", "harreman_group"]
        msi_scores_sample = rna_adata.obsm["msi_module_scores"].loc[df.index]
        metab_hs_sample = metab_hs_adata[
            metab_hs_adata.obs["sample"] == sample].copy()
        common_idx      = metab_hs_sample.obs_names.intersection(df.index)
        df_template     = df.loc[common_idx, ["sample", "msi_module"]].copy()
        msi_scores_null = msi_scores_sample.loc[common_idx]
        observed_metrics = all_results[sample]["perm"]["observed"]
        null_results = harreman_null_test(
            metab_hs_sample,
            df_template=df_template,
            msi_scores=msi_scores_null,
            observed_metrics=observed_metrics,
            n_perm=N_PERM,
            process=process_per_sample.get(sample, "std"),
            n_std=N_STD,
            random_state=RANDOM_STATE,
        )
        print(f"\n  Gene-metabolite null results — {sample}")
        for metric in ["chi2", "ari", "nmi", "mean_max_r"]:
            obs_val  = null_results["observed"][metric]
            null_arr = null_results["null"][metric]
            p_val    = null_results["p_empirical"][metric]
            print(f"    {metric:<12}  obs={obs_val:.4f}  "
                  f"null_median={np.median(null_arr):.4f}  "
                  f"p={p_val:.4f}")
        all_results[sample]["gene_metab_null"] = null_results

    # ── Save results ───────────────────────────────────────────
    with open(ALL_RESULTS_OUT, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nDone. Saved all_results → {ALL_RESULTS_OUT}")


if __name__ == "__main__":
    main()