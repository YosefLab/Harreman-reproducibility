import harreman
import os
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")


BASE_PATH = "/home/projects/nyosef/oier/Harreman_files/Slide_seq_lung"
ADATA_PATH = os.path.join(BASE_PATH, 'h5ads')
DATA_PATH = os.path.join(BASE_PATH, 'data')
PLOTS_PATH = os.path.join(BASE_PATH, 'plots')


adata = sc.read_h5ad(os.path.join(ADATA_PATH, 'Slide_seq_lung_adata.h5ad'))

sample_col = 'sample'
n_genes_expr = 50
genes_to_keep = np.zeros(adata.shape[1], dtype=bool)

for sample in adata.obs[sample_col].unique():
    adata_sample = adata[adata.obs[sample_col] == sample]
    expressed = np.array((adata_sample.X > 0).sum(axis=0)).flatten()
    genes_to_keep |= expressed >= n_genes_expr

adata = adata[:, genes_to_keep]


neighborhood_radius_values = [50, 100, 150, 200, 250, 300, 350, 400]
ref_value = 100

def autocorrelation_robustness_pipeline(adata, neighborhood_radius):
    
    harreman.tl.compute_knn_graph(adata,
                           compute_neighbors_on_key="spatial",
                           neighborhood_radius=neighborhood_radius,
                           weighted_graph=False,
                           sample_key='sample')
    
    n_neighbors = adata.obsp['weights'].sum(axis=1).A1
    
    harreman.hs.compute_local_autocorrelation(adata, layer_key="counts", model='bernoulli', species='human', use_metabolic_genes=True)
    
    gene_autocorrelation_results = adata.uns['gene_autocorrelation_results']
    
    return gene_autocorrelation_results, n_neighbors

gene_autocorrelation_results_dict = {}
n_neighbors_dict = {}
for neighborhood_radius in neighborhood_radius_values:
    print(neighborhood_radius)
    gene_autocorrelation_results, n_neighbors = autocorrelation_robustness_pipeline(adata, neighborhood_radius)
    gene_autocorrelation_results_dict[neighborhood_radius] = gene_autocorrelation_results
    n_neighbors_dict[neighborhood_radius] = n_neighbors

for key, df in gene_autocorrelation_results_dict.items():
    df["Rank"] = range(1, len(df)+1)

C_df = pd.concat(
    {key: df["C"] for key, df in gene_autocorrelation_results_dict.items()}, axis=1
)

C_diff_df = C_df.subtract(C_df.loc[:, ref_value], axis=0)

C_rank_df = pd.concat(
    {key: df["Rank"] for key, df in gene_autocorrelation_results_dict.items()}, axis=1
)
C_diff_melt = C_diff_df.reset_index().melt(
    id_vars="Gene", 
    var_name="Radius", 
    value_name="Value"
)

C_rank_melt = C_rank_df.reset_index().melt(
    id_vars="Gene", 
    var_name="Radius", 
    value_name="Value"
)
C_diff_melt['Rank'] = C_rank_melt['Value']

fig = (
ggplot(data=C_diff_melt, mapping=aes(x="Radius", y="Value", fill="Radius"))
# + geom_boxplot(width=0.8, size=0.3, outlier_size=0.5, varwidth=True)
+ geom_boxplot(outlier_shape="")
+ geom_point(aes(color="Rank"), size=0.5, data=C_diff_melt, show_legend=True)
+ scale_fill_manual(values=["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00"])
+ scale_color_gradientn(
        colors=["#ED3E30", "yellow", "#1C4573"],  # Define multiple colors
        limits=(0, 200))
+ xlab('Neighborhood radius (um)')
+ ylab('Difference in Hotspot Autocorrelation vs. 100 um')
# + ylim(-0.1, 0.1)
+ theme_classic() 
+ theme(plot_title = element_text(hjust = 0.5,
                                margin={"t": 0, "b": 5, "l": 0, "r": 0},
                                size = 14,
                                face='bold'),
        legend_position = "none",
        axis_title_x = element_text(size = 12),
        axis_title_y = element_text(size = 12),
        axis_text_x = element_text(margin={"t": 0, "b": 0, "l": 10, "r": 0}, size = 11),
        axis_text_y = element_text(margin={"t": 0, "b": 0, "l": 0, "r": 10}, size = 11),
        panel_border = element_rect(color='black'),
        panel_background = element_rect(colour = "black",
                                        linewidth = 1),
        figure_size=(6, 4))
)
fig.save(os.path.join(PLOTS_PATH, 'autocor_robustness_box_plot.svg'), format='svg')

n_neighbors_df = pd.DataFrame(
    [(key, value) for key, array in n_neighbors_dict.items() for value in array],
    columns=["Radius", "Value"]
)
n_neighbors_df["Radius"] = n_neighbors_df["Radius"].astype('category')

fig = (
ggplot(data=n_neighbors_df, mapping=aes(x="Radius", y="Value", fill="Radius")) 
# + geom_point(alpha=0.2, color='gray', fill='gray')
+ geom_boxplot(width=0.8, size=0.3, outlier_size=0.2, varwidth=True, fill='lightgrey')
+ xlab('Neighborhood radius (um)')
+ ylab('Number of neighbors')
+ theme_classic() 
+ theme(plot_title = element_text(hjust = 0.5,
                                margin={"t": 0, "b": 5, "l": 0, "r": 0},
                                size = 14,
                                face='bold'),
        legend_position = "none",
        axis_title_x = element_text(size = 12),
        axis_title_y = element_text(size = 12),
        axis_text_x = element_text(margin={"t": 0, "b": 0, "l": 10, "r": 0}, size = 11),
        axis_text_y = element_text(margin={"t": 0, "b": 0, "l": 0, "r": 10}, size = 11),
        panel_border = element_rect(color='black'),
        panel_background = element_rect(colour = "black",
                                        linewidth = 1),
        figure_size=(6, 2))
)
fig.save(os.path.join(PLOTS_PATH, 'autocor_n_neighbors_box_plot.svg'), format='svg')


neighborhood_radius_values = [50, 100, 150, 200, 250, 300, 350, 400]
ref_value = 100

def pairwise_correlation_robustness_pipeline(adata, neighborhood_radius):
    
    harreman.tl.compute_knn_graph(adata,
                           compute_neighbors_on_key="spatial",
                           neighborhood_radius=neighborhood_radius,
                           weighted_graph=False,
                           sample_key='sample')
        
    harreman.hs.compute_local_autocorrelation(adata, layer_key="counts", model='bernoulli', species='human', use_metabolic_genes=True)
    
    gene_autocorrelation_results = adata.uns['gene_autocorrelation_results']
    genes = gene_autocorrelation_results.loc[gene_autocorrelation_results.Z_FDR < 0.01].sort_values('Z', ascending=False).index
    
    harreman.hs.compute_local_correlation(adata, genes=genes)
    
    harreman.hs.create_modules(adata, min_gene_threshold=20)
    
    harreman.hs.calculate_module_scores(adata)
    
    return adata.uns['gene_modules'], adata.obsm['module_scores']

def pair_modules_by_similarity(gene_modules, module_scores, gene_modules_ref, module_scores_ref):
    
    jaccard_sim_matrix = np.zeros((len(gene_modules.keys()), len(gene_modules_ref.keys())))
    for i in range(len(gene_modules.keys())):
        for j in range(len(gene_modules_ref.keys())):
            key_i = list(gene_modules.keys())[i]
            key_j = list(gene_modules_ref.keys())[j]
            a = gene_modules[key_i]
            b = gene_modules_ref[key_j]
            jaccard_sim_matrix[i,j] = len(list(set(a) & set(b))) / len(list(set(a) | set(b)))
    jaccard_sim_matrix_df = pd.DataFrame(jaccard_sim_matrix, index=gene_modules.keys(), columns=gene_modules_ref.keys())
    
    correlation_matrix = pd.DataFrame(
        [[module_scores[col1].corr(module_scores_ref[col2]) for col2 in module_scores_ref.columns] for col1 in module_scores.columns],
        index=module_scores.columns,
        columns=module_scores_ref.columns
    )
    
    similarity_matrix = (jaccard_sim_matrix_df + correlation_matrix)/2
    max_similarity = similarity_matrix.idxmax()
    
    correlation_values = [correlation_matrix.loc[max_similarity[mod], mod] for mod in max_similarity.index]
    jaccard_values = [jaccard_sim_matrix_df.loc[max_similarity[mod], mod] for mod in max_similarity.index]
    
    return correlation_values, jaccard_values

pairwise_correlation_gene_modules_dict = {}
pairwise_correlation_module_scores_dict = {}
for neighborhood_radius in neighborhood_radius_values:
    print(neighborhood_radius)
    gene_modules, module_scores = pairwise_correlation_robustness_pipeline(adata, neighborhood_radius)
    pairwise_correlation_gene_modules_dict[neighborhood_radius] = gene_modules
    pairwise_correlation_module_scores_dict[neighborhood_radius] = module_scores
    
correlation_values_dict = {}
jaccard_values_dict = {}
n_modules_dict = {}
for neighborhood_radius in neighborhood_radius_values:
    print(neighborhood_radius)
    gene_modules = pairwise_correlation_gene_modules_dict[neighborhood_radius]
    module_scores = pairwise_correlation_module_scores_dict[neighborhood_radius]
    gene_modules_ref = pairwise_correlation_gene_modules_dict[ref_value]
    module_scores_ref = pairwise_correlation_module_scores_dict[ref_value]
    correlation_values, jaccard_values = pair_modules_by_similarity(gene_modules, module_scores, gene_modules_ref, module_scores_ref)
    correlation_values_dict[neighborhood_radius] = correlation_values
    jaccard_values_dict[neighborhood_radius] = jaccard_values
    n_modules_dict[neighborhood_radius] = len(gene_modules.keys())

n_modules_df = pd.DataFrame.from_dict(n_modules_dict.items()).rename(columns={0: 'Radius', 1: 'n_modules'})

jaccard_values_df = pd.concat(
    {key: pd.Series(list) for key, list in jaccard_values_dict.items()}, axis=1
).melt( 
    var_name="Radius", 
    value_name="Value"
)

correlation_values_df = pd.concat(
    {key: pd.Series(list) for key, list in correlation_values_dict.items()}, axis=1
).melt( 
    var_name="Radius", 
    value_name="Value"
)

jaccard_values_df["Radius"] = jaccard_values_df["Radius"].astype('category')
correlation_values_df["Radius"] = correlation_values_df["Radius"].astype('category')
n_modules_df["Radius"] = n_modules_df["Radius"].astype('category')

fig = (
ggplot(data=jaccard_values_df, mapping=aes(x="Radius", y="Value", fill="Radius")) 
# + geom_point(alpha=0.2, color='gray', fill='gray')
+ geom_boxplot(width=0.8, outlier_shape="o", outlier_size=0.5)
# + geom_jitter(width = 0.2, size=0.5, alpha=0.6)
# + geom_boxplot(outlier_shape=None)
# + geom_point(size=0.5, alpha=0.6)
+ scale_fill_manual(values=["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6"])
+ xlab('Neighborhood radius (um)')
+ ylab('Max. Jaccard Similarity with Modules from 100 um')
+ theme_classic() 
+ ylim(0,1)
+ theme(plot_title = element_text(hjust = 0.5,
                                margin={"t": 0, "b": 5, "l": 0, "r": 0},
                                size = 14,
                                face='bold'),
        legend_position = "none",
        axis_title_x = element_text(size = 12),
        axis_title_y = element_text(size = 12),
        axis_text_x = element_text(margin={"t": 0, "b": 0, "l": 10, "r": 0}, size = 11),
        axis_text_y = element_text(margin={"t": 0, "b": 0, "l": 0, "r": 10}, size = 11),
        panel_border = element_rect(color='black'),
        panel_background = element_rect(colour = "black",
                                        linewidth = 1),
        figure_size=(6, 4))
)
fig.save(os.path.join(PLOTS_PATH, 'pairwise_cor_jaccard_radius_box_plot_no_points.svg'), format='svg')

fig = (
ggplot(data=correlation_values_df, mapping=aes(x="Radius", y="Value", fill="Radius")) 
# + geom_point(alpha=0.2, color='gray', fill='gray')
+ geom_boxplot(width=0.8, outlier_shape="o", outlier_size=0.5)
# + geom_jitter(width = 0.2, size=0.5, alpha=0.6)
+ scale_fill_manual(values=["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6"])
+ xlab('Neighborhood radius (um)')
+ ylab('Max. Module Correlations with Modules from 100 um')
+ theme_classic() 
+ ylim(-1,1)
+ theme(plot_title = element_text(hjust = 0.5,
                                margin={"t": 0, "b": 5, "l": 0, "r": 0},
                                size = 14,
                                face='bold'),
        legend_position = "none",
        axis_title_x = element_text(size = 12),
        axis_title_y = element_text(size = 12),
        axis_text_x = element_text(margin={"t": 0, "b": 0, "l": 10, "r": 0}, size = 11),
        axis_text_y = element_text(margin={"t": 0, "b": 0, "l": 0, "r": 10}, size = 11),
        panel_border = element_rect(color='black'),
        panel_background = element_rect(colour = "black",
                                        linewidth = 1),
        figure_size=(6, 4))
)
fig.save(os.path.join(PLOTS_PATH, 'pairwise_cor_max_cor_radius_box_plot_no_points.svg'), format='svg')

y_ticks = range(0, n_modules_df['n_modules'].max() + 2, 2)

fig = (
ggplot(data=n_modules_df, mapping=aes(x="Radius", y="n_modules")) 
+ geom_bar(stat='identity', fill='lightgrey') 
+ xlab('Neighborhood radius (um)')
+ ylab('Number of gene modules')
+ scale_y_continuous(breaks=y_ticks)
+ theme_classic() 
+ theme(plot_title = element_text(hjust = 0.5,
                                margin={"t": 0, "b": 5, "l": 0, "r": 0},
                                size = 14,
                                face='bold'),
        legend_position = "none",
        axis_title_x = element_text(size = 12),
        axis_title_y = element_text(size = 12),
        axis_text_x = element_text(margin={"t": 0, "b": 0, "l": 10, "r": 0}, size = 11),
        axis_text_y = element_text(margin={"t": 0, "b": 0, "l": 0, "r": 10}, size = 11),
        panel_border = element_rect(color='black'),
        panel_background = element_rect(colour = "black",
                                        linewidth = 1),
        figure_size=(6, 1.5))
)
fig.save(os.path.join(PLOTS_PATH, 'n_modules_n_neighbors_bar_plot.svg'), format='svg')

print('Finished successfully.')
