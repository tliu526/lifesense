"""
Utilities for processing lifesense data files.
"""

import pandas as pd
import numpy as np
from haversine import haversine
from scipy.signal import lombscargle
from sklearn.cluster import KMeans

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
    df['date'] = pd.to_datetime(df['adj_ts'], unit='s').dt.floor('d')
    df['hour'] = pd.to_datetime(df['adj_ts'], unit='s').dt.floor('H')
    
    return df

def process_cal(id, cal_df):
    """
    Processes pdk-phone-calls dataframes.


    """
    cal_df['adj_call_ts'] = cal_df['call_timestamp']+ (cal_df['timezone-offset']*1000)
    cal_df['call_time'] = pd.to_datetime(cal_df['adj_call_ts'], unit='ms')


def build_cal_hr(pid, loc):
    """
    Builds a df with the following call statistics partitioned by hour:
        - tot_call_count
        - tot_call_duration
        TODO incoming/outgoing

    Args:
        pid (str): the pid to target
    Returns:
        pd.DataFrame
    """
    print(pid)
    cal_df = pd.read_pickle("{}/{}.df".format(loc, pid))
    if cal_df.shape[0] < 1:
        return 
    cal_df = format_time(cal_df)
    
    cal_df['adj_call_ts'] = cal_df['call_timestamp']+ (cal_df['timezone-offset']*1000)
    cal_df['call_time'] = pd.to_datetime(cal_df['adj_call_ts'], unit='ms')
    cal_df['call_hour'] = pd.to_datetime(cal_df['adj_call_ts'], unit='ms').dt.floor('H')
    
    cal_hr = pd.DataFrame()
    call_counts = cal_df.groupby(['hour'])['number'].count()
    #call_counts = cal_df.groupby(['call_hour'])['number'].count()
    cal_hr['tot_call_count'] = call_counts.resample('1H').sum()
    call_dur = cal_df.groupby(['hour'])['duration'].sum()
    #call_dur = cal_df.groupby(['call_hour'])['duration'].sum()
    cal_hr['tot_call_duration'] = call_dur.resample('1H').sum()
    
    cal_hr = cal_hr.reset_index()
    cal_hr['pid'] = pid
    return cal_hr


def build_sms_hr(pid, loc):
    """
    Builds a df with the following sms statistics partitioned by hour:
        - tot_sms_count
        - tot_sms_length
        - in_sms_count
        - in_sms_length
        - out_sms_count
        - out_sms_length

    Args:
        pid (str): the pid to target
    Returns:
        pd.DataFrame
    """
    print(pid)
    sms_df = pd.read_pickle("{}/{}.df".format(loc, pid))
    if sms_df.shape[0] < 1:
        return 
    sms_df = format_time(sms_df)

    sms_hr = pd.DataFrame()
    sms_counts = sms_df.groupby(['hour'])['address'].count()
    sms_hr['tot_sms_count'] = sms_counts.resample('1H').sum()
    sms_dur = sms_df.groupby(['hour'])['length'].sum()
    sms_hr['tot_sms_length'] = sms_dur.resample('1H').sum()

    sms_in = sms_df.loc[sms_df['direction'] == 'incoming']
    sms_counts = sms_in.groupby(['hour'])['address'].count()
    sms_hr['in_sms_count'] = sms_counts.resample('1H').sum()
    sms_dur = sms_in.groupby(['hour'])['length'].sum()
    sms_hr['in_sms_length'] = sms_dur.resample('1H').sum()

    sms_out = sms_df.loc[sms_df['direction'] == 'outgoing']
    sms_counts = sms_out.groupby(['hour'])['address'].count()
    sms_hr['out_sms_count'] = sms_counts.resample('1H').sum()
    sms_dur = sms_out.groupby(['hour'])['length'].sum()
    sms_hr['out_sms_length'] = sms_dur.resample('1H').sum()

    sms_hr = sms_hr.reset_index()
    sms_hr['pid'] = pid
    return sms_hr

def process_fus_daily(fus, cluster_radius=0.2):
    """
    Assumes date, pid columns are populated
    """
    
    # get stationary locations
    fus['prev_lat'] = fus['latitude'].shift()
    fus['prev_long'] = fus['longitude'].shift()
    fus['dist'] = fus.apply(lambda x: haversine((x.latitude, x.longitude), (x.prev_lat, x.prev_long)), axis=1) # in km
    fus['prev_timestamp'] = fus['timestamp'].shift()
    fus['delta_timestamp'] = ((fus['timestamp'] - fus['prev_timestamp']) / (60 * 60)).astype(float) # change to hours
    fus['velocity'] = fus['dist'] / fus['delta_timestamp']
    fus['stationary'] = fus['velocity'] < 1
    fus_stationary = fus[fus['stationary']]
    
    loc_var = np.log(fus_stationary['latitude'].var() + fus_stationary['longitude'].var())

    # assign clusters
    cur_mean = 1
    cur_clusters = 0
    while cur_mean > cluster_radius:
        cur_clusters += 1
        X = fus_stationary[['latitude', 'longitude']]
        kmeans = KMeans(n_clusters=cur_clusters, random_state=0).fit(X)
        transform_X = kmeans.transform(X)
        labels = kmeans.labels_
        clusters = kmeans.cluster_centers_
        X = X.reset_index(drop=True)
        X['labels'] = labels
        X['center'] = X.apply(lambda x: clusters[int(x.labels)], axis=1)
        X['dist'] = X.apply(lambda x: haversine((x.latitude, x.longitude), x.center), axis=1)
        cur_mean = X['dist'].mean()
    
    # get daily entropy
    fus_stationary = fus_stationary.reset_index(drop=True)
    fus_stationary['cluster'] = X['labels']
    label_group = fus_stationary.groupby(['date', 'cluster'])['delta_timestamp'].sum().unstack()
    label_group = label_group.fillna(0)
    label_group['total'] = label_group.sum(axis=1)
    label_group = label_group.div(label_group['total'], axis=0)
    label_group['entropy'] = -(np.log(label_group) * label_group).sum(axis=1)
    label_group = label_group.reset_index()
    
    fus_combined = fus.groupby(['pid', 'date'], as_index=False)['dist'].sum()
    fus_combined = pd.merge(fus_combined, label_group[['date', 'entropy']], on='date', how='outer')
    fus_combined['cluster'] = cur_clusters
    fus_combined['loc_var'] = loc_var
    
    fus_combined['velocity'] = fus.groupby(['pid', 'date'], as_index=False)['velocity'].mean()['velocity']
    #display(fus.groupby(['pid', 'date'], as_index=False)['velocity'].mean())
    #fus_moving = fus[fus['stationary'] > 0]
    
    #fus_combined['transition_time'] = fus_moving.groupby(['pid', 'date'], as_index=False)['delta_timestamp'].sum()
    #display(fus_moving.groupby(['pid', 'date'], as_index=False)['delta_timestamp'].sum())
    
    return fus_combined


def format_raw_fus(fus):
    """Formats raw fus df and returns one ready for processing.

    Args:
        fus (pd.DataFrame): raw fus df pulled from lifesense server

    Returns:
        pd.DataFrame: formatted fus DataFrame
    """
    
    fus = format_time(fus)
    
    final_fus = fus
    final_fus['timestamp'] = final_fus['adj_ts']
    
    return final_fus[['pid', 'longitude', 'latitude', 'timestamp', 'date']]
    

def build_fus(pid, loc):
    fus_df = pd.read_pickle("{}/{}.df".format(loc, pid))
    print(pid)
    if fus_df.shape[0] < 1:
        return 
    print(fus_df.shape)
    fus_df = format_raw_fus(fus_df)
    return process_fus_daily(fus_df)


def get_circadian_movement(fus_df):
    """Calculates the circadian movement based on GPS location for participants
    https://github.com/sosata/CS120DataAnalysis/blob/master/features/estimate_circadian_movement.m
    TODO need to verify the frequency is calculated correctly.
    
    """
    # frequency range of 24 +- 0.5 hrs
    freq = np.linspace(86400-30*60, 86400+30*60, 2*30*60)
    try:
        energy_lat = sum(lombscargle(fus_df['timestamp'], fus_df['latitude'], freq, normalize=True))
        energy_long = sum(lombscargle(fus_df['timestamp'], fus_df['longitude'], freq, normalize=True))
    except ZeroDivisionError:
        return np.nan
    
    tot_energy = energy_lat + energy_long
    if tot_energy > 0:
        return np.log(energy_lat + energy_long)
    else:
        return np.nan


def build_circadian_stats(pid, loc):
    """Gets aggregate daily sms statistics for a given pid
    
    """
    print(pid)
    fus_df = pd.read_pickle("{}/{}.df".format(loc, pid))
    if fus_df.shape[0] < 1:
        return 
    
    fus_df = format_raw_fus(fus_df)
    fus_df["is_wkday"] = (pd.to_datetime(fus_df['date']).dt.dayofweek < 5).astype(float)

    agg_df = pd.DataFrame()
    agg_df['pid'] = [pid]
    agg_df['circ_movt_tot'] = get_circadian_movement(fus_df) 
    agg_df['circ_movt_wkday'] = get_circadian_movement(fus_df.loc[fus_df['is_wkday'] == 1]) 
    agg_df['circ_movt_wkend'] = get_circadian_movement(fus_df.loc[fus_df['is_wkday'] == 0]) 
    return agg_df


apps = [
    'katana',
    'orca', 
    'messaging',
    'launcher',
    'chrome',
    'email',
    'instagram',
    'youtube',
    'maps',
    'snapchat',
    'browser'
]


def process_fga_time(time, fga_group):
    """
    Processes application foreground time within the given screen group.
    
    Assumes fga_group is grouped by some unit of time.
    
    Args:
        time (datetime)
        fga_group (pd.DataFrame)
    
    Returns:
        dict: (k,v) app keyword, time spent 
    """
    app_time = {}
    for app in apps:
        app_time[app] = 0
    app_time['hr'] = time

    idx = 0
    while idx < fga_group.shape[0]:
        cur_app = None
        for app in apps:
            if app in fga_group.iloc[idx]['application']:
                cur_app = app
                break
                
        if cur_app is not None and fga_group.iloc[idx]['screen_active']:
            if idx < (fga_group.shape[0]-1):
                app_time[cur_app] += (fga_group.iloc[idx+1]['time'] -  fga_group.iloc[idx]['time']).total_seconds()
    
            elif idx == fga_group.shape[0]-1:
                # we're in the case where we're at the bottom of the hour
                app_time[cur_app] += ((time + pd.Timedelta(1, unit='h')) -  fga_group.iloc[idx]['time']).total_seconds()
        idx +=1

    return app_time


def build_fga_hr(pid, loc):
    fga_df = pd.read_pickle("{}/{}.df".format(loc, pid))
    if fga_df.shape[0] < 1:
        return 
    fga_df = format_time(fga_df)
    fga_hr = pd.DataFrame()
    fga_slim = fga_df[['hour', 'time', 'screen_active', 'application', 'adj_ts']]
    for time, group in fga_slim.groupby("hour"):
        hr = pd.DataFrame(process_fga_time(time, group), index=[0])
        fga_hr = fga_hr.append(hr)

    fga_hr = fga_hr.set_index('hr')
    fga_hr = fga_hr.resample('1H').sum()
    fga_hr = fga_hr.reset_index()
    fga_hr['pid'] = pid

    return fga_hr