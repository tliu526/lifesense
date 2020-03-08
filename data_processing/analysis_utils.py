"""
LifeSense feature analysis utilities.

Moves correlation analysis present in lifesense_cluster_change_over_time and jama_change_analysis into functions
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler


def build_col_dict(cols, feat_df, seq_df, target, transform_log=False, pre_days=3, post_days=3):
    """Builds a (col, [vals]) dictionary for the given features.
    
    """
    col_dict = {}

    for col in cols:
        col_dict[col] = []

    for idx, row in seq_df.iterrows():
        if pd.isna(row[target]):
            for col in cols:
                col_dict[col].append(np.nan)
            continue
        else:
            date = row['date']
            pid = row['pid']
            sel_df =  feat_df[(feat_df['pid'] == pid) &
                             (feat_df['date'] >= (date.floor('D') - pd.Timedelta(pre_days, unit='D'))) & 
                             (feat_df['date'] <= (date.floor('D') + pd.Timedelta(post_days, unit='D')))]
            sel_mean = sel_df.mean()
            
            if transform_log: sel_mean = np.log(sel_mean + 1)

            for col in cols:
                col_dict[col].append(sel_mean[col])
                
    return col_dict

#%%

def ortho_rotation(components, method='varimax', tol=1e-6, max_iter=100):
    """Return rotated components.
    
    https://github.com/scikit-learn/scikit-learn/pull/11064/files
    """
    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(components, rotation_matrix)
        if method == "varimax":
            tmp = np.diag((comp_rot ** 2).sum(axis=0)) / nrow
            tmp = np.dot(comp_rot, tmp)
        elif method == "quartimax":
            tmp = 0
        u, s, v = np.linalg.svd(
            np.dot(components.T, comp_rot ** 3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    return np.dot(components, rotation_matrix).T


def plot_decomp_components(transformer, n_comps, cols, title):
    """Plots decomposition components"""
    n_cols = len(cols)
    
    sns.heatmap(np.transpose(transformer.components_), annot=True, vmin=-1, vmax=1, cmap="coolwarm")
    plt.ylim(0,n_cols)
    plt.xlim(0,n_comps)
    plt.yticks(np.arange(0.5,n_cols), cols, rotation='horizontal')
    plt.xticks(np.arange(0.5, n_comps), np.arange(1, n_comps+1))
    plt.xlabel("Components")
    plt.title(title)
    plt.show()


def plot_varimax(transformer, n_comps, cols, title):
    """Plots varimax rotation"""
    varimax = ortho_rotation(transformer.components_.T)
    n_cols = len(cols)

    plt.rcParams["figure.figsize"] = [20,12]
    sns.heatmap(np.transpose(varimax), annot=True, vmin=-1, vmax=1, cmap="coolwarm")
    plt.ylim(0,n_cols)
    plt.xlim(0,n_comps)
    plt.yticks(np.arange(0.5,n_cols), cols, rotation='horizontal')
    plt.xticks(np.arange(0.5, n_comps), np.arange(1, n_comps+1))
    plt.xlabel("Components")
    plt.title(title)
    plt.show()
    

def get_var_explained(transformer):
    var_df = pd.Series(transformer.explained_variance_ratio_).to_frame()
    var_df.columns = ['var_explained']
    return var_df


def generate_decomp_features(feat_df, feat_cols, id_cols, n_comps, agg_name, method="pca", suffix="agg", random_state=0):
    """Generates aggregate decomp features for the given columns. 

    Returns the aggregate feature columns as well as feature transformer.

    Params:
        feat_df (pd.DataFrame): feature frame to perform PCA across
        feat_cols (list): a list of columns to generate decomposition
        id_cols (list): a list of columns for Pandas merge indexing
        n_comps (int): number of decomposition components to target
        agg_name (str): the aggregate column name
        method (str): whether to run factor analysis ("factor") or PCA ("pca")
        suffix (str): column suffix for the generated decomp columns
        random_state (int): random seed for fast SVD approximations of PCA/FA
    
    Returns:
        pd.DataFrame, sklearn.decomposition
    """
    feat_df = feat_df.copy() 
    feat_df[feat_cols] = StandardScaler().fit_transform(feat_df[feat_cols])
    feat_df = feat_df.dropna(how='any', subset=feat_cols)
    feat_df = feat_df.reset_index(drop=True)

    decomp_method = None
    if method == "pca":
        decomp_method = PCA
    elif method == "factor":
        decomp_method = FactorAnalysis

    transformer = decomp_method(n_components=n_comps, random_state=random_state)
    transformed_features = transformer.fit_transform(feat_df[feat_cols])

    decomp_cols = ["{}_{}{}_{}".format(agg_name, method, i, suffix) for i in range(1, n_comps + 1)]
    decomp_df = pd.DataFrame(transformed_features, columns=decomp_cols)
    
 
    decomp_df = pd.concat([feat_df[id_cols], decomp_df], axis=1)

    return decomp_df, transformer

def generate_unit_features(feat_df, include_cols):
    """

    """
    pass

def generate_corr_shift_df(shift_df, col_suffix):
    """

    """
    pass


def build_simple_corr(corr_df, target, method='pearson', padjust='fdr_bh', pval=0.05):
    """
    Builds simple correlation DataFrame from corr_df of the target survey.


    Params:
        corr_type (str): which correlations to compute. If 'both', then both spearman and pearson correlations are displayed
    """
    pass


def build_partial_corr(corr_df, target, covar, method='pearson', padjust='fdr_bh', pval=0.05, covar_name=None):
    """
    Builds partial correlation DataFrame from corr_df of the target survey, controlling for covar.
    
    corr_df (pd.DataFrame): correlation frame, assuming each row is an observation
    target (str): targe column, can be a string prefix or suffix
    covar (list): a list of covariates to control for
    covar_name (str): optional name for covariates in the display
    """
    
    partial_corr = pg.pairwise_corr(data=corr_df, covar=covar, method=method)
    _, p_adj = pg.multicomp(partial_corr['p-unc'].values, alpha=pval, method=padjust)
    partial_corr['p-corr'] = p_adj
    
    partial_corr = partial_corr.loc[(partial_corr['p-corr'] < pval) & (~partial_corr['X'].str.contains(target)) & (partial_corr['Y'].str.contains(target))]
    partial_corr['r_ctl'] = partial_corr['r']
    partial_corr['p_ctl'] = partial_corr['p-corr']
    if covar_name is not None:
        partial_corr['covar'] = covar_name
    
    
    partial_corr = partial_corr[['X', 'Y', 'covar', 'r_ctl', 'p_ctl']]
    
    # drop the controlling covars for the raw pairwise correlation
    pairwise_corr = pg.pairwise_corr(data=corr_df.drop(covar, axis='columns'), method=method, padjust=padjust)
    pairwise_corr['r_unctl'] = pairwise_corr['r']
    pairwise_corr['p_unctl'] = pairwise_corr['p-corr']

    partial_corr = partial_corr.merge(pairwise_corr[['X', 'Y', 'r_unctl', 'p_unctl', 'n']], on=['X', 'Y'], how='left').sort_values('p_ctl')
    return partial_corr.style.set_caption(method)


def generate_corr_analysis():
    """
    Wrapper method for running a full analysis pipeline

    """
    pass


# test bed
#%%
test_df = pd.read_pickle("~/lifesense/dig_state_df_test.df")
display(test_df.head())
fga_cols = ['katana', 'orca', 'messaging', 'email', 'instagram', 'youtube', 'maps', 'snapchat', 'browser', 'chrome']
df, transformer = generate_decomp_features(test_df, fga_cols, ['pid', 'study_wk'], 5, 'fga')
display(df.head())
print(df.shape)
plot_decomp_components(transformer, 5, fga_cols, "PCA of App Features")
display(df.tail())

if __name__ == '__main__':
    pass


# %%
