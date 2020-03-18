#%% [markdown] 
# Script for aggregating the raw data. 
# Only needs to be run once due to chunking of raw data pulls, not part of 
# general pipeline.

#%% imports
import pandas as pd
import pickle

#%% constants
target_loc = "/data/tliu/wave1"
id_loc = "/home/tliu/lifesense/data_pull/ids/wave1_ids.txt"
sensors_loc = "/home/tliu/lifesense/wave1_sensors.txt"

wave1_ids = []
with open(id_loc, "r") as ids_f:
    for line in ids_f.readlines():
        wave1_ids.append(line.strip())

sensors = []
with open(sensors_loc, "r") as sensors_f:
    for line in sensors_f.readlines():
        sensors.append(line.strip())

print(len(wave1_ids))
print(wave1_ids)
print(sensors)

#%% check the "lifesense data directory"
test_id = "00746649"

def format_time(df):
    """
    Takes timestamp and timezone-offset to create time columns.

    Args:
        df (pd.DataFrame)

    Returns:
        df with adjusted time columns:
            - adj_ts: timestamp (s) with offset
            - time: adjusted time
            - date: adjusted time, day 
            - hour: adjusted time, hour
    """
    df['timestamp'] = df['timestamp'].astype(int)
    df['timezone-offset'] = df['timezone-offset'].astype(int)

    df['adj_ts'] = df['timestamp'] + df['timezone-offset']
    df['time'] = pd.to_datetime(df['adj_ts'], unit='s')
    df['date'] = pd.to_datetime(df['adj_ts'], unit='s').dt.round('d')
    df['hour'] = pd.to_datetime(df['adj_ts'], unit='s').dt.floor('H')
    
    return df

#%%
test_loc = pd.read_pickle("/home/tliu/lifesense/data_pull/pdk-location/{}.df".format(test_id))
test_loc = format_time(test_loc)
test_loc['timestamp'].tail(10)

#%%
test_loc = pd.read_pickle("/data/tliu/wk4_ls_data/pdk-location/{}.df".format(test_id))
test_loc = format_time(test_loc)
test_loc['timestamp'].head(10)

#%% [markdown]
# Steps to aggregate data:
# 1. Pull from "data_pull" directories, exclude the overlap in dates
# 2. append all data sources, including the gaps
# 3. reindex and sort by timestamp

#%% Merge data pull and wk 4 data
wk1_loc = pd.read_pickle("/home/tliu/lifesense/data_pull/pdk-location/{}.df".format(test_id))
wk4_loc = pd.read_pickle("/data/tliu/wk4_ls_data/pdk-location/{}.df".format(test_id))

wk1_loc['source'] = 'wk1'
wk4_loc['source'] = 'wk4'

min_timestamp = wk4_loc['timestamp'].min()
wk1_loc = wk1_loc[wk1_loc['timestamp'] < min_timestamp]

combined_loc = wk1_loc.append(wk4_loc)
combined_loc = combined_loc.reset_index(drop=True)
combined_loc = combined_loc.sort_values(by='timestamp')
combined_loc.shape

#%% merge wk1 wk4 function
def merge_wk1_wk4(pid, generator):

    wk1_path = "/home/tliu/lifesense/data_pull/{}/{}.df".format(generator, pid)
    wk4_path = "/data/tliu/wk4_ls_data/{}/{}.df".format(generator, pid)

    wk1_df = pd.read_pickle(wk1_path)
    wk4_df = pd.read_pickle(wk4_path)
    combined_df = wk1_df 
    if combined_df.shape[0] > 0:
        if wk4_df.shape[0] > 0:
            min_timestamp = wk4_df['timestamp'].min()
            wk1_df = wk1_df[wk1_df['timestamp'] < min_timestamp]

            combined_df = wk1_df.append(wk4_df)
            combined_df = combined_df.sort_values(by='timestamp')
    return combined_df

test = merge_wk1_wk4(test_id, 'pdk-location')
test.shape

#%% check gaps
test_gap = pd.read_pickle("/data/tliu/gaps/wk4/pdk-location/{}.df".format(test_id))
test_gap['source'] = 'wk4_gap'
gap_min = test_gap['timestamp'].min()
gap_max = test_gap['timestamp'].max()

wk7_loc = pd.read_pickle("/data/tliu/wk7_ls_data/pdk-location/{}.df".format(test_id))
wk7_loc['source'] = 'wk7'
combined_loc = combined_loc.append(test_gap)
combined_loc = combined_loc.append(wk7_loc)
combined_loc = combined_loc.sort_values(by=['timestamp'])
combined_loc = combined_loc.reset_index(drop=True)

min_idx = combined_loc.index[combined_loc['timestamp'] == gap_min].tolist()[0]
max_idx = combined_loc.index[combined_loc['timestamp'] == gap_max].tolist()[0]

#display(combined_loc.loc[min_idx-5:, ['timestamp', 'source']])
#display(combined_loc.loc[max_idx-5:, ['timestamp', 'source']])
# Gaps look to be filled properly, with no overlap

#%% merge_all function
def merge_all(pid, generator):
    combined_df = merge_wk1_wk4(pid, generator)

    gap_path = "/data/tliu/gaps/wk{}/{}/{}.df"
    wk_path = "/data/tliu/wk{}_ls_data/{}/{}.df"
    
    # merge wk 4 gaps
    wk4_gaps = pd.read_pickle(gap_path.format(4, generator, pid))
    combined_df = combined_df.append(wk4_gaps)

    remaining_wks = [7,10,13,16]
    for wk in remaining_wks:
        wk_df = pd.read_pickle(wk_path.format(wk, generator, pid))
        if wk_df.shape[0] > 0:
            combined_df = combined_df.append(wk_df)

        gap_df = pd.read_pickle(gap_path.format(wk, generator, pid))
        if gap_df.shape[0] > 0:
            combined_df = combined_df.append(gap_df)    

    combined_df = combined_df.reset_index(drop=True)
    return combined_df

#%% test dumping to file
"""
for pid in wave1_ids[:1]:
    print(pid)
    for generator in sensors:
        print(generator)
        df = merge_all(pid, generator)
        write_path = target_loc + "/{}/{}.df".format(generator, pid)
        df.to_pickle(write_path)
"""

#%% final function
def aggregate_sensors(pid):
    print(pid)
    for generator in sensors:
        df = merge_all(pid, generator)
        write_path = target_loc + "/{}/{}.df".format(generator, pid)
        df.to_pickle(write_path)

if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(processes=4) as pool:
        pool.map(aggregate_sensors, wave1_ids)



