"""
Harreman × MSI Validation Pipeline
====================================
Improvements over the original code:

1. Module merging is now criterion-based (spatial score correlation >= threshold)
   rather than manually specified, removing potential circularity / cherry-picking.

2. Zone comparison uses BOTH hard-assignment (chi-squared) AND soft/continuous
   comparison (Spearman correlation between module score vectors), reducing
   sensitivity to arbitrary argmax assignments.

3. The full permutation null distribution is now properly computed and compared
   to the observed statistic, with an empirical p-value reported for BOTH metrics.

4. Binarization diagnostics: active-spot fraction is reported per metabolite,
   and the threshold sensitivity is checked across n_std values (0.5, 1, 1.5, 2).

5. Cell-type-based zonation is added as an alternative baseline, so the
   Harreman-specific contribution can be assessed relative to a naive comparator.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import harreman

from scipy.stats import zscore, spearmanr
from scipy.stats.contingency import chi2_contingency
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import liana as li
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 0.  Paths (edit as needed)
# ─────────────────────────────────────────────
ADATA_PATH = "..."
DATA_PATH  = "..."


# ═══════════════════════════════════════════════════════════
# SECTION 1 ─ Preprocessing helpers
# ═══════════════════════════════════════════════════════════

def standardize_msi(adata, log_transform=True):
    X = np.asarray(adata.X)
    if log_transform:
        X = np.log1p(X)
    X_scaled = zscore(X, axis=0, nan_policy="omit")
    X_scaled = np.nan_to_num(X_scaled)
    adata.layers["msi_zscore"] = X_scaled
    return adata


def robust_scale_msi(adata):
    X = np.asarray(adata.X)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad[mad == 0] = 1
    X_scaled = (X - med) / mad
    adata.layers["msi_robust"] = X_scaled
    return adata


def feature_filtering(adata, sample_col="sample", min_frac=0.05, var_quantile=0.1):
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


# ═══════════════════════════════════════════════════════════
# SECTION 2 ─ Binarization with diagnostics
# ═══════════════════════════════════════════════════════════

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
    """Mean + n_std * SD threshold binarization."""
    means = np.nanmean(X, axis=0)
    stds  = np.nanstd(X,  axis=0)
    thresholds = means + n_std * stds
    return (X >= thresholds).astype(float)


def binarize_and_report(X, var_names, process="std", n_std=1.0,
                        verbose=True):
    """
    Binarize a matrix and report active-spot fractions per feature.

    Returns
    -------
    X_bin   : np.ndarray  (binarized)
    report  : pd.DataFrame with columns ['feature', 'active_frac']
    """
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


def threshold_sensitivity_check(X, var_names, n_std_values=(0.5, 1.0, 1.5, 2.0)):
    """
    Run binarization at several n_std thresholds and report active-spot
    fraction statistics. Useful for choosing a robust threshold.
    """
    rows = []
    for n in n_std_values:
        X_bin = std_binarize_matrix(X, n_std=n)
        af = X_bin.mean(axis=0)
        rows.append({
            "n_std": n,
            "median_active_frac": np.median(af),
            "mean_active_frac":   np.mean(af),
            "pct_below_1pct":     (af < 0.01).mean() * 100,
            "pct_above_50pct":    (af > 0.50).mean() * 100,
        })
    return pd.DataFrame(rows)


def msi_scores_from_adata(adata, process="std", n_std=1.0):
    scores = np.asarray(adata.layers["raw"])
    X_bin, report = binarize_and_report(scores, adata.var_names,
                                        process=process, n_std=n_std)
    return pd.DataFrame(X_bin, index=adata.obs_names,
                        columns=adata.var_names), report


def harreman_scores_from_adata(adata, process="std", n_std=1.0):
    scores = adata.uns["interacting_cell_results"]["np"]["m"]["cs"].copy()
    X_bin, report = binarize_and_report(scores, adata.uns["metabolites"],
                                        process=process, n_std=n_std)
    return pd.DataFrame(X_bin, index=adata.obs_names,
                        columns=adata.uns["metabolites"]), report


def harreman_sample_scores_from_adata(adata_, obs_names, process="std",
                                      n_std=1.0):
    scores = adata_.uns["interacting_cell_results"]["np"]["m"]["cs"].copy()
    idx = [list(obs_names).index(name) for name in adata_.obs_names]
    scores = scores[idx]
    X_bin, report = binarize_and_report(scores, adata_.uns["metabolites"],
                                        process=process, n_std=n_std)
    return pd.DataFrame(X_bin, index=adata_.obs_names,
                        columns=adata_.uns["metabolites"]), report


# ═══════════════════════════════════════════════════════════
# SECTION 3 ─ Criterion-based super-module merging
#             (replaces the hard-coded sample_super_module_dict)
# ═══════════════════════════════════════════════════════════

def merge_modules_by_correlation(adata, corr_threshold=0.8,
                                 score_key="module_scores"):
    """
    Merge fine-grained Harreman/MSI modules whose spatial score vectors
    have Spearman correlation >= corr_threshold.

    Parameters
    ----------
    adata          : AnnData with obsm[score_key] already computed
    corr_threshold : minimum Spearman r to merge two modules
    score_key      : key in adata.obsm containing the module scores

    Returns
    -------
    super_module_dict : dict  {super_id (int) : [original_module_ids]}
    super_scores      : pd.DataFrame (spots × super-modules)
    """
    scores = adata.obsm[score_key]
    module_names = list(scores.columns)
    n = len(module_names)

    # Pairwise Spearman correlation matrix
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(scores.iloc[:, i], scores.iloc[:, j])
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    # Greedy merging: build connected components where r >= threshold
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
        sm_id + 1: [module_names[idx] for idx in group]
        for sm_id, group in enumerate(groups)
    }

    # Average scores within each super-module
    super_scores = pd.DataFrame(
        {f"SM{sm_id}": scores[members].mean(axis=1)
         for sm_id, members in super_module_dict.items()},
        index=scores.index,
    )

    print(f"[merge_modules] {n} modules → {len(super_module_dict)} super-modules "
          f"(threshold r={corr_threshold})")
    for sm_id, members in super_module_dict.items():
        print(f"  SM{sm_id}: {members}")

    return super_module_dict, super_scores, corr_matrix


# ═══════════════════════════════════════════════════════════
# SECTION 4 ─ Zonation pipelines
# ═══════════════════════════════════════════════════════════

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

    # Criterion-based super-module merging
    super_module_dict, super_scores, corr_matrix = merge_modules_by_correlation(
        adata_, corr_threshold=corr_threshold)
    adata_.obsm["super_module_scores"] = super_scores
    adata_.uns["gene_modules_sm"] = super_module_dict
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

    # Hard assignment (kept for chi-squared)
    adata_.obs["top_group"] = (
        pd.DataFrame(zscore(adata_.obsm["module_scores"], axis=0),
                     index=adata_.obsm["module_scores"].index,
                     columns=adata_.obsm["module_scores"].columns)
        .idxmax(axis=1)
    )
    return adata_


# ═══════════════════════════════════════════════════════════
# SECTION 5 ─ Zone-comparison metrics
#             (hard assignment + soft/continuous)
# ═══════════════════════════════════════════════════════════

def compute_chi2_stat(df):
    """Chi-squared statistic from a contingency table."""
    contingency = pd.crosstab(df["msi_module"], df["harreman_group"])
    chi2, _, _, _ = chi2_contingency(contingency)
    return chi2


def compute_soft_overlap(msi_scores, harreman_scores):
    """
    Soft/continuous comparison: average Spearman correlation between
    matched super-module score vectors.

    Each MSI super-module score vector is correlated with every Harreman
    module score vector. We take the maximum per MSI module and average
    across modules.

    Parameters
    ----------
    msi_scores      : pd.DataFrame (spots × MSI super-modules)
    harreman_scores : pd.DataFrame (spots × Harreman modules)

    Returns
    -------
    mean_max_r : float  (higher = better alignment)
    r_matrix   : pd.DataFrame of pairwise Spearman r values
    """
    common_idx = msi_scores.index.intersection(harreman_scores.index)
    A = msi_scores.loc[common_idx]
    B = harreman_scores.loc[common_idx]

    r_matrix = pd.DataFrame(
        index=A.columns, columns=B.columns, dtype=float)

    for col_a in A.columns:
        for col_b in B.columns:
            r, _ = spearmanr(A[col_a], B[col_b])
            r_matrix.loc[col_a, col_b] = r

    mean_max_r = r_matrix.max(axis=1).mean()
    return mean_max_r, r_matrix


def compute_ari_nmi(df):
    """Adjusted Rand Index and NMI between hard zone assignments."""
    ari = adjusted_rand_score(df["msi_module"], df["harreman_group"])
    nmi = normalized_mutual_info_score(df["msi_module"], df["harreman_group"])
    return ari, nmi


def compute_all_metrics(df, msi_scores, harreman_scores):
    """
    Compute all overlap metrics between MSI and Harreman zonations.

    Returns a dict with keys: chi2, ari, nmi, mean_max_r
    """
    chi2 = compute_chi2_stat(df)
    ari, nmi = compute_ari_nmi(df)
    mean_max_r, r_matrix = compute_soft_overlap(msi_scores, harreman_scores)
    return {"chi2": chi2, "ari": ari, "nmi": nmi,
            "mean_max_r": mean_max_r, "r_matrix": r_matrix}


# ═══════════════════════════════════════════════════════════
# SECTION 6 ─ Permutation test (label shuffling)
# ═══════════════════════════════════════════════════════════

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


def permutation_test(df, msi_scores, harreman_scores,
                     n_perm=1000, random_state=0):
    """
    Permutation test for ALL metrics (chi2, ARI, NMI, mean_max_r).

    For mean_max_r, harreman_scores columns are shuffled (column permutation)
    rather than spot labels, which is the correct null for a continuous metric.

    Returns
    -------
    dict with keys:
        observed      : dict of observed metric values
        null          : dict of null distributions (np.ndarray per metric)
        p_empirical   : dict of empirical p-values
    """
    rng = np.random.default_rng(random_state)

    # Observed
    obs = compute_all_metrics(df, msi_scores, harreman_scores)

    null = {"chi2": [], "ari": [], "nmi": [], "mean_max_r": []}

    for _ in range(n_perm):
        # Hard-assignment metrics: shuffle spot labels within sample
        df_perm = shuffle_labels_within_sample(
            df, label_col="harreman_group", sample_col="sample", rng=rng)

        null["chi2"].append(compute_chi2_stat(df_perm))
        ari_p, nmi_p = compute_ari_nmi(df_perm)
        null["ari"].append(ari_p)
        null["nmi"].append(nmi_p)

        # Soft metric: permute Harreman module score columns
        harreman_perm = harreman_scores.copy()
        col_order = rng.permutation(harreman_scores.columns)
        harreman_perm.columns = col_order
        mmr_p, _ = compute_soft_overlap(msi_scores, harreman_perm)
        null["mean_max_r"].append(mmr_p)

    null = {k: np.array(v) for k, v in null.items()}

    p_empirical = {
        k: np.mean(null[k] >= obs[k])
        for k in ["chi2", "ari", "nmi", "mean_max_r"]
    }

    return {"observed": obs, "null": null, "p_empirical": p_empirical}


def summarize_permutation_results(results, sample_name=""):
    """Pretty-print permutation test results."""
    print(f"\n{'='*55}")
    print(f"  Permutation test results  {sample_name}")
    print(f"{'='*55}")
    for metric in ["chi2", "ari", "nmi", "mean_max_r"]:
        obs = results["observed"][metric]
        null = results["null"][metric]
        p   = results["p_empirical"][metric]
        print(f"  {metric:<12}  obs={obs:.4f}  "
              f"null_median={np.median(null):.4f}  p={p:.4f}")
    print()


# ═══════════════════════════════════════════════════════════
# SECTION 7 ─ Cell-type baseline
# ═══════════════════════════════════════════════════════════

def celltype_baseline_zonation(rna_adata, celltype_col="cell_type",
                               n_clusters=None):
    """
    Build a naive cell-type-composition-based zone assignment.
    Each spot is assigned to the zone defined by its dominant cell type
    (or KMeans on cell-type proportion vectors for deconvolved data).

    Parameters
    ----------
    rna_adata     : AnnData with obs[celltype_col] (categorical) OR
                    obsm['cell_type_proportions'] (deconvolved)
    celltype_col  : obs column for discrete cell-type labels
    n_clusters    : if set, use KMeans on cell-type proportions instead

    Returns
    -------
    pd.Series of zone labels, indexed by obs_names
    """
    if "cell_type_proportions" in rna_adata.obsm and n_clusters is not None:
        props = rna_adata.obsm["cell_type_proportions"]
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        labels = km.fit_predict(props)
        return pd.Series(
            [f"CT_zone_{l}" for l in labels],
            index=rna_adata.obs_names, name="ct_zone")

    if celltype_col in rna_adata.obs:
        return rna_adata.obs[celltype_col].astype(str).rename("ct_zone")

    raise ValueError(
        f"Neither 'cell_type_proportions' in obsm nor '{celltype_col}' in obs.")


def evaluate_baseline(df_template, msi_scores, rna_adata,
                      celltype_col="cell_type", n_clusters=None):
    """
    Compute all overlap metrics for the cell-type baseline.

    Returns the same dict format as compute_all_metrics.
    """
    ct_labels = celltype_baseline_zonation(
        rna_adata, celltype_col=celltype_col, n_clusters=n_clusters)

    df_ct = df_template.copy()
    df_ct["harreman_group"] = ct_labels.loc[df_ct.index].values

    # For soft metric, build dummy one-hot scores per CT zone
    ct_dummies = pd.get_dummies(ct_labels).loc[
        msi_scores.index.intersection(ct_labels.index)]

    return compute_all_metrics(df_ct, msi_scores, ct_dummies)


# ═══════════════════════════════════════════════════════════
# SECTION 8 ─ Gene-metabolite shuffling nulls
# ═══════════════════════════════════════════════════════════

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
    kmeans = KMeans(n_clusters=n_bins, random_state=random_state,
                    n_init="auto")
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


def shuffle_gene_metabolite_matrix_full(gene_metab_df, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    shuffled = gene_metab_df.copy()
    for metab in shuffled.columns:
        values = shuffled[metab].values.copy()
        rng.shuffle(values)
        shuffled[metab] = values
    return shuffled


# ═══════════════════════════════════════════════════════════
# SECTION 9 ─ Full Harreman runner (for gene-metab null)
# ═══════════════════════════════════════════════════════════

def run_harreman(adata, shuffled, process="std", n_std=1.0):
    """
    Run the full Harreman pipeline with a (potentially shuffled)
    gene-metabolite database. Returns spot-level top_group labels
    AND the module score DataFrame.
    """
    harreman.pp.extract_interaction_db(adata, species="human",
                                       database="both")
    adata.varm["database"] = shuffled

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
        verbose=True)

    scores_df, _ = harreman_scores_from_adata(adata, process=process,
                                              n_std=n_std)
    metab_scores_adata = ad.AnnData(scores_df)
    metab_scores_adata.obs["sample"] = adata.obs["sample"]
    metab_scores_adata.obsm["spatial"] = adata.obsm["spatial"]

    harreman.tl.compute_knn_graph(metab_scores_adata,
                                  compute_neighbors_on_key="spatial",
                                  n_neighbors=5,
                                  weighted_graph=False,
                                  sample_key="sample")
    harreman.hs.compute_local_autocorrelation(
        metab_scores_adata, model="bernoulli")
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
                       n_perm=100, process="std", n_std=1.0,
                       random_state=0):
    """
    Gene-metabolite null: shuffle the transporter-metabolite database
    and re-run Harreman, measuring overlap with MSI zones at each iteration.

    This tests whether the real gene-metabolite associations are necessary
    to produce the observed overlap, beyond what random transporter sets give.

    Returns
    -------
    dict with keys: observed, null, p_empirical  (same format as permutation_test)
    """
    rng = np.random.default_rng(random_state)
    gene_metab_df = adata.varm["database"]

    # Observed (real database)
    obs_labels, obs_scores = run_harreman(adata, gene_metab_df,
                                          process=process, n_std=n_std)
    if obs_labels is None:
        raise RuntimeError("Harreman returned no modules on real data.")

    df_obs = df_template.copy()
    df_obs["harreman_group"] = obs_labels.loc[df_obs.index].values
    obs_metrics = compute_all_metrics(df_obs, msi_scores, obs_scores)

    null = {"chi2": [], "ari": [], "nmi": [], "mean_max_r": []}

    for i in range(n_perm):
        shuffled = shuffle_gene_metabolite_matrix_full(gene_metab_df, rng)
        perm_labels, perm_scores = run_harreman(
            adata.copy(), shuffled, process=process, n_std=n_std)

        if perm_labels is None:
            continue

        df_perm = df_template.copy()
        df_perm["harreman_group"] = perm_labels.loc[df_perm.index].values

        m = compute_all_metrics(df_perm, msi_scores, perm_scores)
        for k in ["chi2", "ari", "nmi", "mean_max_r"]:
            null[k].append(m[k])

    null = {k: np.array(v) for k, v in null.items()}
    p_empirical = {
        k: np.mean(null[k] >= obs_metrics[k])
        for k in ["chi2", "ari", "nmi", "mean_max_r"]
    }
    return {"observed": obs_metrics, "null": null, "p_empirical": p_empirical}


# ═══════════════════════════════════════════════════════════
# SECTION 10 ─ Main pipeline
# ═══════════════════════════════════════════════════════════

def main():
    process = "std"
    n_std   = 1.0
    bandwidth = 500
    cutoff    = 0.1

    # ── Load data ──────────────────────────────────────────
    adata    = harreman.read_h5ad(
        os.path.join(ADATA_PATH, "Visium_RCC_Harreman_unrolled.h5ad"))
    msi_neg  = sc.read_h5ad(os.path.join(ADATA_PATH, "neg_msi_all.h5ad"))
    rna      = sc.read_h5ad(os.path.join(ADATA_PATH, "neg_rna.h5ad"))
    metab_hs_adata = harreman.read_h5ad(
        os.path.join(ADATA_PATH, "Visium_RCC_Metabolic_Hotspot.h5ad"))

    # ── Feature filtering ──────────────────────────────────
    msi_neg = feature_filtering(msi_neg)

    metabolite_annotation = pd.read_excel(
        os.path.join(DATA_PATH, "spatial_metabolites_annotation.xlsx"),
        index_col=0).dropna(subset="Metabolites")
    metabolite_annotation["mz"] = metabolite_annotation["mz"].round(3)
    msi_neg.var = (
        msi_neg.var.reset_index()
        .merge(metabolite_annotation, on="mz", how="left")
        .set_index("index")
    )

    # ── Binarization + sensitivity check ──────────────────
    print("\n[Threshold sensitivity check]")
    sensitivity_report = threshold_sensitivity_check(
        np.asarray(msi_neg.layers["raw"]), msi_neg.var_names)
    print(sensitivity_report.to_string(index=False))

    msi_scores_df, msi_bin_report = msi_scores_from_adata(
        msi_neg, process=process, n_std=n_std)
    msi_metab_scores_adata = ad.AnnData(msi_scores_df)
    msi_metab_scores_adata.obs["sample"]   = msi_neg.obs["sample"]
    msi_metab_scores_adata.obsm["spatial"] = msi_neg.obsm["spatial"]

    # ── MSI zonation (per sample, criterion-based merging) ─
    msi_adatas = {}
    for sample in rna.obs["sample"].unique():
        print(f"\n[MSI zonation] {sample}")
        s_adata = msi_metab_scores_adata[
            msi_metab_scores_adata.obs["sample"] == sample].copy()
        s_adata = run_per_sample_msi_harreman_pipeline(
            s_adata, corr_threshold=0.8)
        msi_adatas[sample] = s_adata

    # ── Interpolate MSI super-modules onto RNA spots ───────
    rna_adatas = {}
    for sample, sample_adata in msi_adatas.items():
        print(f"\n[MSI→RNA interpolation] {sample}")
        sample_rna = rna[rna.obs["sample"] == sample].copy()
        reference  = sample_adata.obsm["spatial"]

        li.utils.spatial_neighbors(
            sample_rna, bandwidth=bandwidth, cutoff=cutoff,
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

    # ── Harreman zonation (per sample) ─────────────────────
    harreman_adatas = {}
    for sample in rna_adatas:
        print(f"\n[Harreman zonation] {sample}")
        sample_adata = adata[adata.obs["sample"] == sample].copy()
        scores_df, _ = harreman_sample_scores_from_adata(
            sample_adata, adata.obs_names, process=process, n_std=n_std)

        metab_scores_adata = ad.AnnData(scores_df)
        metab_scores_adata.obs["sample"]   = sample_adata.obs["sample"]
        metab_scores_adata.obsm["spatial"] = sample_adata.obsm["spatial"]

        metab_scores_adata = run_per_sample_harreman_pipeline(
            metab_scores_adata)
        harreman_adatas[sample] = metab_scores_adata

    # ── Per-sample comparison ──────────────────────────────
    all_results = {}
    for sample, rna_adata in rna_adatas.items():
        print(f"\n{'#'*55}\n  Sample: {sample}\n{'#'*55}")

        harreman_adata = harreman_adatas[sample]
        msi_labels     = rna_adata.obs["top_super_module"]
        harreman_labels= harreman_adata.obs["top_group"]
        sample_labels  = harreman_adata.obs["sample"]

        df = pd.concat(
            [sample_labels[msi_labels.index],
             msi_labels,
             harreman_labels.loc[msi_labels.index]], axis=1)
        df.columns = ["sample", "msi_module", "harreman_group"]

        msi_scores_sample = rna_adata.obsm["msi_module_scores"].loc[df.index]
        har_scores_sample = harreman_adata.obsm["module_scores"].loc[df.index]

        # ── (a) Permutation test (label-shuffling) ─────────
        perm_results = permutation_test(
            df, msi_scores_sample, har_scores_sample,
            n_perm=1000, random_state=42)
        summarize_permutation_results(perm_results, sample_name=sample)

        # ── (b) Cell-type baseline ─────────────────────────
        print(f"[Cell-type baseline] {sample}")
        ct_metrics = evaluate_baseline(
            df, msi_scores_sample, rna_adata,
            celltype_col="cell_type")
        print(f"  chi2={ct_metrics['chi2']:.4f}  "
              f"ari={ct_metrics['ari']:.4f}  "
              f"nmi={ct_metrics['nmi']:.4f}  "
              f"mean_max_r={ct_metrics['mean_max_r']:.4f}")
        print(f"  (compare with Harreman observed: "
              f"chi2={perm_results['observed']['chi2']:.4f}  "
              f"ari={perm_results['observed']['ari']:.4f}  "
              f"nmi={perm_results['observed']['nmi']:.4f}  "
              f"mean_max_r={perm_results['observed']['mean_max_r']:.4f})")

        # ── (c) Residual analysis ──────────────────────────
        contingency_df = pd.crosstab(df["msi_module"], df["harreman_group"])
        chi2, _, _, expected = chi2_contingency(contingency_df.values)
        residuals = (contingency_df.values - expected) / np.sqrt(expected)
        residuals_df = pd.DataFrame(
            residuals,
            index=contingency_df.index,
            columns=contingency_df.columns)
        top_assoc = (residuals_df.stack().reset_index()
                     .rename(columns={"level_0": "MSI_module",
                                      "level_1": "Harreman_group",
                                      0:         "residual"})
                     .query("residual > 0")
                     .sort_values("residual", ascending=False)
                     .head(5))
        print("\nTop MSI ↔ Harreman associations (standardized residuals):")
        print(top_assoc.to_string(index=False))

        all_results[sample] = {
            "perm": perm_results,
            "ct_baseline": ct_metrics,
            "residuals": residuals_df,
        }

    return all_results, rna_adatas, harreman_adatas, msi_adatas


if __name__ == "__main__":
    results, rna_adatas, harreman_adatas, msi_adatas = main()
