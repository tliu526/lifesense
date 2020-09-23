"""
Interactive python script for data missingness in the LifeSense data dumps
"""
#%% [markdown]
# # LifeSense Data Missingness

#%% 
# imports and constants
%matplotlib inline
import json
import pickle

import gmaps
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

baseline_nans = baseline_df.isnull().sum(axis=0)
baseline_nans = baseline_nans[baseline_nans > 0]
count_barplot(baseline_nans, 
              "Baseline missing counts", 
              "survey fields", 
              "Missing/NaN counts")

""" plt.figure(figsize=(16,10))
wk4_nans = wk4_df.isnull().sum(axis=0)
wk4_nans = wk4_nans[wk4_nans > 0]
count_barplot(wk4_nans, 
              "wk4 missing counts", 
              "survey fields", 
              "Missing/NaN counts")
 """

wk4_nans = wk4_df.isnull().sum(axis=0)
wk4_nans = wk4_nans[wk4_nans > 0]
display(wk4_nans.to_frame())

#%% [markdown]
"""
### Missingness Notes
 
#### Baseline missingness
- ext_feedback missing: what is the role of this question?
- 3 participants with missing work schedule information
- other missing fields are sensible (optional, free-text, etc)

#### Week 4 missingness
- 1 participant with missing work schedule information
- no responses on travel question (not forced choice?)
- no smartphone install/reinstall info (can be inferred?)
- 3 participants report problems with app
- no participants report changing sensor settings
- no participants report sleep disturbances
- issues with PSQI aggregation? (psqi_total)
- one participant with missing withdraw response
""" 
#%% 
# prefer not to answer responses

baseline_df = baseline_df.replace(to_replace=99, value=999)
wk4_df = wk4_df.replace(to_replace=99, value=999)

baseline_no_ans = baseline_df.isin([999, 99]).sum(axis=0)
baseline_no_ans = baseline_no_ans[baseline_no_ans > 0]
display(baseline_no_ans)
baseline_no_ans = baseline_df.isin([999, 99]).sum(axis=1)
baseline_no_ans = baseline_no_ans[baseline_no_ans > 0]
display(baseline_no_ans)

wk4_no_ans = wk4_df.isin([999, 99]).sum(axis=0)
wk4_no_ans = wk4_no_ans[wk4_no_ans > 0]
display(wk4_no_ans)
wk4_no_ans = wk4_df.isin([999, 99]).sum(axis=1)
wk4_no_ans = wk4_no_ans[wk4_no_ans > 0]
display(wk4_no_ans)

p0 = wk4_df.iloc[0,:]
display(p0[p0.isin([999,99])])

p6 = wk4_df.iloc[6,:]
display(p6[p6.isin([999,99])])


#%% [markdown]
"""
### "Prefer no answer" Notes

#### Baseline
- one participant had any PNA (prefer no answer) responses
    - no answer on "Have you seen a therapist..."
    - no answer on AUDIT

#### Week 4
- two participants had PNA responses
    - one participant had no answer on "I can recover from mistakes quickly..."
    - one participant had no answer on number of jobs, student/non-student, psqi 08 
"""

#%% [markdown]
# ## PDK Analysis

#%%
# load ids and data

with open("data_pull/test_wave_ids.txt", "r") as testwave_f:
    testwave_ids = [line.strip() for line in testwave_f.readlines()]

with open("data_pull/internal_staff_ids.txt", "r") as internal_f:
    internal_ids = [line.strip() for  line in internal_f.readlines()]

API_KEY = "AIzaSyB4KK750CZGbxfIPUHN-DK4g67QPhv1T6w"

#%%
# sandbox

test_pid = '16784865'
morn_df = pd.read_pickle("data_pull/morning_ema/{}.df".format(test_pid))
even_df = pd.read_pickle("data_pull/evening_ema/{}.df".format(test_pid))
loc_df = pd.read_pickle("data_pull/pdk-location/{}.df".format(test_pid))
morn_df.isna().sum(axis=0)
display(morn_df['for_yesterday'])
display(morn_df.head())
display(loc_df.head())

#%%
# Gmaps exploration
from ipywidgets.embed import embed_minimal_html


gmaps.configure(api_key=API_KEY)
locations = loc_df[['latitude', 'longitude']]
fig = gmaps.figure()
fig.add_layer(gmaps.heatmap_layer(locations))
#embed_minimal_html('export.html', views=[fig])

# TODO test out gmaps API
# can use symbol_layer as per tutorial to map out labelled locations



#%% [markdown]
"""
### Morning/evening EMAs
"""

#%% 


# 

testwave_morn = pd.DataFrame()