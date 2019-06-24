#%% [markdown]
# # PDK Client Data Pull
# Notebook for PDK client manipulation

#%% 
# imports and constants

import json
import pickle

import numpy as np
import pandas as pd

from pdk_client import PDKClient
from IPython.display import display, HTML

SITE_URL = "https://lifesense.fsm.northwestern.edu/data"
TOKEN = "oCZUlrEtmlJRsy9b8fOpNTthlMPT7kU6GcniFLt0yLH4Yz0ExkGeflPuaPlOwCyj"
PAGE_SIZE = 100

#%%
# build the client obj and initial query

client = PDKClient(site_url=SITE_URL, token=TOKEN)
query = client.query_data_points(page_size=PAGE_SIZE)
query.count()

#%% test cell for location
location_query = query.filter(source='23853441', generator_identifier='pdk-location').order_by('created')

first_loc_point = location_query.first()
loc_df = pd.DataFrame()
for loc in location_query:
    loc_df = loc_df.append(pd.read_json(json.dumps(loc)).iloc[0])
    break

loc_df['pid'] = '23853441'
print(loc_df.shape)
display(loc_df.head())

#%%
# pull and dump locations for testwave ids 
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
# pull and dump locations for internal staff ids
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

#%% test cell for evening EMA processing
id = '70859397'

ema_query = query.filter(source=id, generator_identifier='evening_phq8').order_by('created')

point = ema_query.first()
ema_df = pd.DataFrame()

ema_df = ema_df.append(pd.DataFrame.from_dict(point).iloc[0])

id = '23853441'
ema_query = query.filter(source=id, generator_identifier='evening_phq8').order_by('created')

point = ema_query.first()
ema_df = ema_df.append(pd.DataFrame.from_dict(point).iloc[0])

display(ema_df[ema_df.columns[ema_df.columns.str.startswith('place')]].head())

ema_df.columns

# TODO columns will be variable depending on the number of places a participant
# has visited in a given day, how to handle this?
# place-:
# kind, latitude, longitude, name, other, with-others, 


#%% test cell for morning EMA processing
id = '70859397'

ema_query = query.filter(source=id, generator_identifier='morning_phq8').order_by('created')

point = ema_query.first()
ema_df = pd.DataFrame()

ema_df = ema_df.append(pd.DataFrame.from_dict(point).iloc[0])

display(ema_df)



#%% 
# process morning/evening EMA for testwave

testwave_ids = []
with open("data_pull/test_wave_ids.txt", "rb") as testwave_f:
    for line in testwave_f.readlines():
        testwave_ids.append(line.strip())

#generators = ['morning_phq8', 'evening_phq8']
generators = ['morning_ema', 'evening_ema']
for id in testwave_ids:
    print(id)
    
    for gen_id in generators:
        ema_query = query.filter(source=id, generator_identifier=gen_id).order_by('created')
        
        ema_df = pd.DataFrame()
        for point in ema_query:
            ema_df = ema_df.append(pd.DataFrame.from_dict(point).iloc[0])

        ema_df['pid'] = id 
        ema_df['data_source'] = 'test_wave'
        ema_df = ema_df.reset_index(drop=True)
        print(ema_df.shape)
        #display(ema_df.head())
        pickle.dump(ema_df, open("data_pull/{}/{}.df".format(gen_id, id), 'wb'), -1)

#%% 
# process morning/evening EMA for internal testers

internal_ids = []
with open("data_pull/internal_staff_ids.txt", "rb") as testwave_f:
    for line in testwave_f.readlines():
        internal_ids.append(line.strip())

#generators = ['morning_phq8', 'evening_phq8']
generators = ['morning_ema', 'evening_ema']
for id in internal_ids:
    print(id)
    
    for gen_id in generators:
        ema_query = query.filter(source=id, generator_identifier=gen_id).order_by('created')
        
        ema_df = pd.DataFrame()
        for point in ema_query:
            ema_df = ema_df.append(pd.DataFrame.from_dict(point).iloc[0])

        ema_df['pid'] = id 
        ema_df['data_source'] = 'test_wave'
        ema_df = ema_df.reset_index(drop=True)
        print(ema_df.shape)
        #display(ema_df.head())
        pickle.dump(ema_df, open("data_pull/{}/{}.df".format(gen_id, id), 'wb'), -1)
