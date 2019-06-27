"""
Interactive python script for data missingness in the LifeSense data dumps
"""
#%% [markdown]
# # LifeSense Data Missingness
# Notebook for PDK client manipulation


#%% 
# imports and constants

import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from IPython.display import display, HTML

#%% [markdown]
# ## Plotting functions

#%%
def count_barplot(series, title, xlabel, ylabel):
    """Plots the counts in the series as a barplot."""
    series = series.to_frame()
    series = series.reset_index()
    sns.barplot(x="index", y=0, data=series)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.show()

#%% [markdown]
# ## RedCap Analysis

#%%
baseline_df = pd.read_excel("data_pull/LS_TestWave_SC_BL_WK4_Data_060719.xlsx",
                            sheet_name=0)
wk4_df = pd.read_excel("data_pull/LS_TestWave_SC_BL_WK4_Data_060719.xlsx",
                       sheet_name=2)
display(baseline_df.head())
display(wk4_df)

#%%
# NaNs/missingness
baseline_df = baseline_df.replace(to_replace=99, value=999)
wk4_df = wk4_df.replace(to_replace=99, value=999)

baseline_nans = baseline_df.isnull().sum(axis=0)
baseline_nans = baseline_nans[baseline_nans > 0]
count_barplot(baseline_nans, 
              "Baseline missing counts", 
              "survey fields", 
              "Missing/NaN counts")

plt.figure(figsize=(16,10))
wk4_nans = wk4_df.isnull().sum(axis=0)
wk4_nans = wk4_nans[wk4_nans > 0]
count_barplot(wk4_nans, 
              "wk4 missing counts", 
              "survey fields", 
              "Missing/NaN counts")
