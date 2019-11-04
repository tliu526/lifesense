"""
Utilities for processing lifesense data files.
"""

import pandas as pd
import numpy as np
from haversine import haversine
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


def build_cal_hr(pid):
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
    cal_df = pd.read_pickle("data_pull/pdk-phone-calls/{}.df".format(pid))
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


def build_sms_hr(pid):
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
    sms_df = pd.read_pickle("data_pull/pdk-text-messages/{}.df".format(pid))
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