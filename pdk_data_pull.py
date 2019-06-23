#%% [markdown]
# # PDK Client Data Pull
# Notebook for PDK client manipulation

#%%
%matplotlib inline

import json
import pickle

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pdk_client import PDKClient

SITE_URL = "https://lifesense.fsm.northwestern.edu/data"
TOKEN = "oCZUlrEtmlJRsy9b8fOpNTthlMPT7kU6GcniFLt0yLH4Yz0ExkGeflPuaPlOwCyj"
PAGE_SIZE = 100
#%%
client = PDKClient(site_url=SITE_URL, token=TOKEN)
query = client.query_data_points(page_size=PAGE_SIZE)
query.count()

#%%
location_query = query.filter(source='23853441', generator_identifier='pdk-location').order_by('created')

first_loc_point = location_query.first()
loc_df = pd.DataFrame()
for loc in location_query:
    loc_df = loc_df.append(pd.read_json(json.dumps(loc)).iloc[0])

loc_df['pid'] = '23853441'
print(loc_df.shape)
display(loc_df.head())

#%%
# pull and dump testwave ids
testwave_ids = []
with open("data_pull/test_wave_ids.txt", "rb") as testwave_f:
    for line in testwave_f.readlines():
        testwave_ids.append(line.strip())

for id in testwave_ids:
    print(id)
    location_query = query.filter(source=id, generator_identifier='pdk-location').order_by('created')
    
    loc_df = pd.DataFrame()
    for loc in location_query:
        loc_df = loc_df.append(pd.read_json(json.dumps(loc)).iloc[0])

    loc_df['pid'] = id 
    loc_df['data_source'] = 'test_wave'
    print(loc_df.shape)
    pickle.dump(loc_df, open("data_pull/loc/{}.df".format(id), 'wb'), -1)

#%%
# pull and dump internal staff ids
testwave_ids = []
with open("data_pull/internal_staff_ids.txt", "rb") as testwave_f:
    for line in testwave_f.readlines():
        testwave_ids.append(line.strip())

for id in testwave_ids:
    print(id)
    location_query = query.filter(source=id, generator_identifier='pdk-location').order_by('created')
    
    loc_df = pd.DataFrame()
    for loc in location_query:
        loc_df = loc_df.append(pd.read_json(json.dumps(loc)).iloc[0])

    loc_df['pid'] = id 
    loc_df['data_source'] = 'internal_staff'
    print(loc_df.shape)
    pickle.dump(loc_df, open("data_pull/loc/{}.df".format(id), 'wb'), -1)
