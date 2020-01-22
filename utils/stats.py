"""
Functions for descriptive statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_ind
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as robjects
import rpy2
from rpy2.robjects.packages import importr

# import rpy2 modules
utils = importr('utils')
lmtest = importr('lmtest')
Hmisc = importr("Hmisc")
pandas2ri.activate()


def build_corr_mat(corrs, p_vals, labels, title, alpha, figsize=[20,12], vmin=-1, vmax=1):
    """
    returns the matplotlib plt object for the specified correlations.
    """
    plt.rcParams["figure.figsize"] = figsize
    plt.imshow(corrs, vmin=vmin, vmax=vmax)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = "{0:.2f}".format(corrs[i, j])
            p = p_vals[i,j]
            if p < alpha:
                text = text + "*"
            plt.text(j,i, text, ha="center", va="center", color="w")
    plt.xticks([x for x in range(len(labels))], labels, rotation=45, ha="right", rotation_mode='anchor')
    plt.yticks([x for x in range(len(labels))], labels)
    plt.xlim(-0.5, corrs.shape[1] - .5)
    plt.ylim(-0.5, corrs.shape[1] - .5)
    plt.colorbar()
    plt.title(title)
    return plt


def run_r_corr(df, corr_type='spearman', p_correction='BH'):
    """
    Runs R correlation calculations and p-value corrections on the given dataframe.
    
    :returns: a tuple of (correlations, counts, p_values)
    """
    num_cols = len(df.columns.values)
    r_dataframe = pandas2ri.py2ri(df)
    r_as = r['as.matrix']
    rcorr = r['rcorr'] 
    r_p_adjust = r['p.adjust']
    result = rcorr(r_as(r_dataframe), type=corr_type)
    rho = result[0]
    n = result[1]
    p = result[2]
    
    if p_correction is not None:
        p = r_p_adjust(p, p_correction)
    r_corrs = pandas2ri.ri2py(rho)
    r_p_vals = pandas2ri.ri2py(p)
    r_counts = pandas2ri.ri2py(n)
    r_p_vals = np.reshape(r_p_vals, (num_cols,num_cols))
    return r_corrs, r_counts, r_p_vals


def build_ttest_dfs(pid_df, group_col, val_cols):
    """
    Runs t-tests on the given pid_df DataFrame against the groupings as defined in group_col.
    
    :param pid_df: a DataFrame aggregated by participant pids
    :param group_col: the column name to group on
    :param val_cols: the columns we want to run t-tests on
    :returns: t_df, p_df DataFrames containing the t and p values
    """
    index = pd.Index(data=pid_df[group_col].unique(), name=group_col).sort_values()
    t_df = pd.DataFrame(index = index, columns = val_cols)
    p_df = pd.DataFrame(index = index, columns = val_cols)

    for group in pid_df[group_col].unique():
        selected_group = pid_df[pid_df[group_col] == group]
        rest_group =  pid_df[pid_df[group_col] != group]
        for col in val_cols:
            t, p = ttest_ind(selected_group[col].values, 
                             rest_group[col].values, 
                             nan_policy='omit')

            t_df.loc[group][col] = t
            p_df.loc[group][col] = p
    
    for col in p_df.columns.values:
            t_df[col] = t_df[col].apply(lambda x: format(float(x), '.3f'))
            p_df[col] = p_df[col].apply(lambda x: format(float(x), '.3f'))
    return t_df, p_df


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