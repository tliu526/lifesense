
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

#%% test cell for filtered query object

id = '29878406'
client = PDKClient(site_url=SITE_URL, token=TOKEN)
query = client.query_data_points(page_size=PAGE_SIZE, source=id)
query.count()

#%% test cell for evening EMA processing
ema_query = query.filter( generator_identifier='evening_ema').order_by('created')
ema_df = pd.DataFrame()
point = ema_query.first()
ema_df = ema_df.append(pd.DataFrame.from_dict(point).iloc[0])

display(ema_df[ema_df.columns[ema_df.columns.str.startswith('place')]].head())

ema_df.head()

#%% test cell for morning EMA processing
ema_query = query.filter(generator_identifier='morning_ema').order_by('created')

point = ema_query.first()
ema_df = pd.DataFrame()

ema_df = ema_df.append(pd.DataFrame.from_dict(point).iloc[0])

metadata_dict = point['passive-data-metadata'] 

metadata_df = pd.Series(point['passive-data-metadata']).to_frame().transpose()
ema_df.reset_index(inplace=True, drop=True)
full_df = pd.concat([metadata_df, ema_df], axis=1)
full_df.drop('passive-data-metadata', axis='columns', inplace=True)
display(full_df)

#%%
# processing functions

def process_generators(ids, generators, data_source):
    """Processes the data for the given generators and ids.

    Dumps the resulting DataFrames into pickled files.

    Args:
        query: a PDKPointQuery object, TODO fix
        ids: list of ids to query
        generators: list of generators to pull
        data_source: str indicating which group the ids belong to

    """
    for id in ids:
        print(id)
        query = client.query_data_points(page_size=PAGE_SIZE, source=id)
        for gen_id in generators:
            ema_query = query.filter(source=id, generator_identifier=gen_id).order_by('created')
            
            ema_df = pd.DataFrame()
            for point in ema_query:
                point_df = pd.DataFrame.from_dict(point).iloc[0].to_frame().transpose()
                metadata_df = pd.Series(point['passive-data-metadata']).to_frame().transpose()
                point_df.reset_index(inplace=True, drop=True)
                point_df = pd.concat([metadata_df, point_df], axis=1, sort=True)
                
                point_df.drop('passive-data-metadata', axis='columns', inplace=True)

                ema_df = ema_df.append(point_df)

            ema_df['pid'] = id 
            ema_df['data_source'] = data_source
            ema_df = ema_df.reset_index(drop=True)
            print(ema_df.shape)
            #display(ema_df.head())
            pickle.dump(ema_df, open("data_pull/{}/{}.df".format(gen_id, id), 'wb'), -1)


#%% 
# process morning/evening EMA for wave 1

testwave_ids = []
with open("data_pull/wave1_ids.txt", "rb") as testwave_f:
    for line in testwave_f.readlines():
        testwave_ids.append(line.strip())

generators = ['morning_ema', 'evening_ema', 'morning_phq8', 'evening_phq8']
process_generators(testwave_ids, generators, 'wave1')

#%% test cell for location
location_query = query.filter(generator_identifier='pdk-location').order_by('created')

first_loc_point = location_query.first()
loc_df = pd.DataFrame()
for loc in location_query:
    loc_df = loc_df.append(pd.read_json(json.dumps(loc)).iloc[0])
    break

print(loc_df.shape)
display(loc_df.head())

#%% pull and dump locations for wave 1 ids
testwave_ids = []
with open("data_pull/wave1_ids.txt", "rb") as testwave_f:
    for line in testwave_f.readlines():
        testwave_ids.append(line.strip())

for id in testwave_ids:
    print(id)
    query = client.query_data_points(page_size=PAGE_SIZE, source=id)
    location_query = query.filter(source=id, generator_identifier='pdk-location').order_by('created')
    
    loc_df = pd.DataFrame()
    for point in location_query:
        point_df = pd.DataFrame.from_dict(point).iloc[0].to_frame().transpose()
        metadata_df = pd.Series(point['passive-data-metadata']).to_frame().transpose()
        metadata_df = metadata_df.drop(['latitude', 'longitude'], axis='columns')
        point_df.reset_index(inplace=True, drop=True)
        point_df = pd.concat([metadata_df, point_df], axis=1, sort=True)
        
        point_df.drop('passive-data-metadata', axis='columns', inplace=True)
        #print("pre missing cols:{}".format(point_df.shape[1]))
        missing_cols = [col for col in loc_df.columns.values if col not in point_df.columns.values]
        #print(missing_cols)
        
        if len(missing_cols) > 0 and loc_df.shape[0] > 0:
            for col in missing_cols:
                point_df[col] = np.nan
            point_df = point_df[loc_df.columns]
        
        #print("post-missing cols:{}".format(point_df.shape[1]))

        
        loc_df = loc_df.append(point_df)

        
    loc_df['pid'] = id 
    loc_df['data_source'] = 'wave1'
    print(loc_df.shape)
    #display(loc_df.head())
    
    pickle.dump(loc_df, open("data_pull/pdk-location/{}.df".format(id), 'wb'), -1)

#%% test cell for communication
screen_query = query.filter(generator_identifier='pdk-text-messages').order_by('created')

point = screen_query.first()
screen_df = pd.DataFrame()

screen_df = screen_df.append(pd.DataFrame.from_dict(point).iloc[0])

metadata_dict = point['passive-data-metadata'] 

metadata_df = pd.Series(point['passive-data-metadata']).to_frame().transpose()
screen_df.reset_index(inplace=True, drop=True)
full_df = pd.concat([metadata_df, screen_df], axis=1)
full_df.drop('passive-data-metadata', axis='columns', inplace=True)
display(full_df)
full_df.shape

#%% process communication and screen state
generators = ['pdk-phone-calls', 'pdk-text-messages', 'pdk-screen-state']
process_generators(testwave_ids, generators, 'wave1')

#%% try filtering based on date
from psycopg2.extras import DateRange, DateTimeTZRange, NumericRange, Range
ema_query = query.filter(generator_identifier='pdk-location',
                         created__gte='2019-08-19',
                         created__lte='2019-08-25').order_by('created')

point = ema_query.first()
count = 0
for p in ema_query:
    count += 1

print(count)
print(ema_query.count())
print(point)
ema_df = pd.DataFrame()

ema_df = ema_df.append(pd.DataFrame.from_dict(point).iloc[0])

metadata_dict = point['passive-data-metadata'] 

metadata_df = pd.Series(point['passive-data-metadata']).to_frame().transpose()
ema_df.reset_index(inplace=True, drop=True)
full_df = pd.concat([metadata_df, ema_df], axis=1)
full_df.drop('passive-data-metadata', axis='columns', inplace=True)
display(full_df)

#%% get counts for a particular time range

generators = [
    'pdk-system-status',
    'pdk-sensor-accelerometer',
    'pdk-device-battery',
    'pdk-location',
    'pdk-time-of-day',
    'pdk-app-event',
    'pdk-foreground-application',
    'pdk-sensor-light',
    'pdk-screen-state',
    'pdk-text-messages',
    'pdk-phone-calls',
    'pdk-google-awareness',
    'evening_phq8',
    'morning_phq8',
    'evening_ema', 
    'morning_ema', 
    'pdk-sensor-step-count'
]

def get_data_counts(id, generators, start_date, end_date):
    """Gets the number of data points collected over the specified id, generator, date range.

    Args:
        id (str): the participant id
        generators (list): the generator names
        start_date (str): the start date of the filter, in yyyy-mm-dd form
        start_date (str): the end date of the filter, in yyyy-mm-dd form

    Returns:
        dict: (generator, count) pairs over the time period
    """
    query = client.query_data_points(page_size=PAGE_SIZE, source=id)
    count_dict = {}
    for generator in generators:
        print(generator)
        gen_query = query.filter(generator_identifier=generator,
                            created__gte=start_date,
                            created__lt=end_date)
        count_dict[generator] = gen_query.count()

    return count_dict
#%% test get_data_counts()
%%time

from datetime import date, timedelta

id = '29878406'
id_d = {}
id_df = pd.DataFrame()

start_date = date(2019, 8, 25)
end_date = date(2019, 8, 27)
cur_date = start_date
while cur_date <= end_date:
    start_str = cur_date.strftime("%Y-%m-%d")
    end_str = (cur_date + timedelta(days=1)).strftime("%Y-%m-%d")
    d = get_data_counts(id, generators, start_str, end_str)
    df = pd.DataFrame(d, index=[0])
    df['date'] = cur_date
    df['pid'] = id
    id_df = id_df.append(df)
    id_d[cur_date.strftime("%Y-%m-%d")] = d
    cur_date += timedelta(days=1) 

#%%

def process_counts(id):
    """Processes counts over the currently hardcoded dates

    """
    id_df = pd.DataFrame()

    start_date = date(2019, 8, 25)
    end_date = date(2019, 8, 27)
    cur_date = start_date
    while cur_date <= end_date:
        start_str = cur_date.strftime("%Y-%m-%d")
        end_str = (cur_date + timedelta(days=1)).strftime("%Y-%m-%d")
        d = get_data_counts(id, generators, start_str, end_str)
        df = pd.DataFrame(d, index=[0])
        df['date'] = cur_date
        df['pid'] = id
        id_df = id_df.append(df)
        cur_date += timedelta(days=1) 

    pickle.dump(id_df, open("data_pull/generator_counts/{}.df".format(id), 'wb'), -1)

process_counts(id)