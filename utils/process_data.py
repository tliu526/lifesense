"""
A collection of data processing utilities for the CS120 dataset.

"""

import os
import datetime


from haversine import haversine
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans

#### CONSTANTS ####

base_path = "CS120/CS120-sensor-csvs/"
id_dirs = os.listdir(base_path)

csv_files = {
    "wtr": "./CS120/CS120Weather/{}/wtr.csv",
    "act": "./CS120/CS120-sensor-csvs/{}/act.csv",
    "app": "./CS120/CS120-sensor-csvs/{}/app.csv",
    "cal": "./CS120/CS120-sensor-csvs/{}/cal.csv",
    "coe": "./CS120/CS120-sensor-csvs/{}/coe.csv",
    "fus": "./CS120/CS120-sensor-csvs/{}/fus.csv",
    "run": "./CS120/CS120-sensor-csvs/{}/run.csv",
    "scr": "./CS120/CS120-sensor-csvs/{}/scr.csv",
    "tch": "./CS120/CS120-sensor-csvs/{}/tch.csv",
    "emm": "./CS120/CS120-sensor-csvs/{}/emm.csv",
    "ems": "./CS120/CS120-sensor-csvs/{}/ems.csv",
    "emc": "./CS120/CS120-sensor-csvs/{}/emc.csv",
    "eml": "./CS120/CS120-sensor-csvs/{}/eml.csv",
    "aud": "./CS120/CS120-sensor-csvs/{}/aud.csv",
}

csv_headers = {
    "wtr": ["timestamp", "temperature", "humidity", "dew_point", "wind_speed", "visibility", "pressure", "windchill", "precipitation", "sky_condition", 'fog', "rain", "snow", "hail", "thunder", "tornado"],
    "emm": ["timestamp", "stress", "mood", "energy", "focus"],
    "ems": ["timestamp", "bed_time", "sleep_time", "wake_time", "up_time", "sleep_quality", "day_type"],
    "act": ["timestamp", "activity_type", "confidence"],
    "app": ["timestamp", "app_package", "app_name", "app_category"],
    "cal": ["timestamp", "call_state"],
    "coe": ["timestamp", "contact_name", "contact_number", "comm_type", "comm_direction"],
    "fus": ["timestamp", "latitude", "longitude", "altitude", "accuracy"],
    "run": ["timestamp", "app_package", "app_category", "task_stack_index"],
    "scr": ["timestamp", "screen_state"],
    "tch": ["timestamp", "last_touch_delay", "touch_count"],
    "emc": ["timestamp", "utc_time", "contact_name", "contact_number", "contact_type", "q1_want", "q2_talk", "q3_loan", "q4_closeness"],
    "eml": ["timestamp", "utc_time", "latitude", "longitude", "radius", "place_name", "place_type", "visit_reason", "accomplishment", 	"pleasure" ],
    "aud": ["timestamp", "power", "frequency", "magnitude"]
}

def convert_col_to_day(df, time_col, day_col, unit='ms'):
    """
    Converts the given time_col (in epoch time) to a new day_col column.
    """
    df[time_col] = pd.to_datetime(df[time_col], unit=unit)
    df[day_col] = pd.DatetimeIndex(df[time_col]).normalize()
    return df


def get_screener_df():
    """
    reads and builds headers for the initial screening csv file
    """
    screener_df = pd.read_csv('CS120/CS120Clinical/CS120Final_Screener.csv', encoding = "ISO-8859-1")
    screener_df = screener_df[['ID', 'score_PHQ', 'score_GAD', 'CONTROL', 'ANXIOUS', 'DEPRESSED', 'DEPRESSED_ANXIOUS']]
    
    group_cols = ['CONTROL', 'ANXIOUS', 'DEPRESSED', 'DEPRESSED_ANXIOUS'] 
    screener_df[group_cols] = screener_df[group_cols].fillna(0)
    screener_df = screener_df.rename(index=str, columns={'ID': 'pid'})
    
    return screener_df


def process_date_daily(df, pid):
    """
    Adds both a 'pid' column and 'date' column, where 'date' is aggregated to day.
    Assumes a 'timestamp' column.

    :param df: pandas df
    :param pid: the pid to insert
    :returns: the modified df
    """
    df.loc[:, 'pid'] = pd.Series([pid for x in range(len(df['timestamp']))])
    df = convert_col_to_day(df, 'timestamp', 'date', unit='s')
    
    return df


def process_date_hourly(df, pid, freq="8H"):
    """
    Adds both a 'pid' column and 'date' column, where 'date' is aggregated 
    to the nearest frequency hour (default 8 hrs).

    Assumes a 'timestamp' column.
    :param df: pandas df
    :param pid: the pid to insert
    :returns: the modified df
    """
    df.loc[:, 'pid'] = pd.Series([pid for x in range(len(df['timestamp']))])
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['date'].dt.round(freq)
    return df

def process_date_windows(df, pid, win_list):
    """
    Adds both a 'pid' column and 'date' column, where 'date' is aggregated 
    within the list of windows, and mapped to end_date. 
    
    Assumes a 'timestamp' column.

    :param df: pandas df
    :param pid: the pid to insert
    :param win_list: a list of (start_time, end_time) tuples
    :returns: the modified df
    """
    df.loc[:, 'pid'] = pd.Series([pid for x in range(len(df['timestamp']))])
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')

    win_df = pd.DataFrame()

    mask = df['pid'] != pid # init mask of Falses
    
    end_times = []
    prev_end_time = datetime.datetime(1985,1,1) # needed in case prev end_time > next start_time
    for start_time, end_time in win_list:
        win_mask = (df['date'] > max(start_time, prev_end_time)) & (df['date'] < end_time)
        mask = mask | win_mask
        end_times.extend(([end_time] * np.sum(win_mask.values)))
        prev_end_time = end_time
        
    win_df = df.loc[mask]
    win_df['date'] = end_times

    return win_df
    

def build_emm_win_dict(window_size=120):
    """
    For each pid, build a list of pairs of (start_time, end_time) of windows before EMM survey.
    
    :param window_size: the size of the window in minutes, default 120
    :returns: a dict with (pid, [(start_time, end_time)]) tuples
    """
    csv_name = 'emm'
    win_dict = {}
    for pid in id_dirs:
        csv_path = csv_files[csv_name].format(pid)
        if os.path.exists(csv_path):
            emm_df = pd.read_csv(csv_path, sep='\t', header=None, names=csv_headers[csv_name])
            emm_df['end_win'] = pd.to_datetime(emm_df['timestamp'], unit='s')
            emm_df['start_win'] = emm_df['end_win'] - pd.DateOffset(minutes=window_size)
            win_dict[pid] = list(zip(emm_df.start_win, emm_df.end_win))
    return win_dict


def z_score(df, in_col):
    """
    computes the z-score of the given column
    """
    df[in_col] = (df[in_col] - df[in_col].mean()) / df[in_col].std()
    return df


def process_csv_all_window(csv_name, window_size=120, **kwargs):
    """
    Aggregates all of the windowed dataframes for the given csv name into a single dataframe.
    """
    df_all = pd.DataFrame(columns=['pid'] + csv_headers[csv_name])
    win_dict = build_emm_win_dict(window_size=window_size)
    for pid in id_dirs:
        csv_path = csv_files[csv_name].format(pid)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep='\t', header=None, 
                              names=csv_headers[csv_name])
            if pid in win_dict:
                df = process_date_windows(df, pid, win_dict[pid]) # adds pid, date columns
                process_func = globals()['process_{}'.format(csv_name)]
                df_window = pd.DataFrame(process_func(df, pid, **kwargs))
                df_all = df_all.append(df_window)
    df_all = df_all.dropna(axis=1, how='all')
    return df_all


def process_csv_all_daily(csv_name, **kwargs):
    """
    Aggregates all of the daily dataframes for the given csv name into a single dataframe.
    """
    df_all = pd.DataFrame(columns=['pid'] + csv_headers[csv_name])

    for pid in id_dirs:
        csv_path = csv_files[csv_name].format(pid)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep='\t', header=None, 
                              names=csv_headers[csv_name])
            df = process_date_daily(df, pid) # adds pid, date columns
            daily_process_func = globals()['process_{}'.format(csv_name)]
            df_day = pd.DataFrame(daily_process_func(df, pid, **kwargs))
            df_all = df_all.append(df_day)
    df_all = df_all.dropna(axis=1, how='all')
    return df_all


def process_csv_all_hourly(csv_name, freq="8H", **kwargs):
    """
    Aggregates all of the daily dataframes for the given csv name into a single dataframe.
    """
    df_all = pd.DataFrame(columns=['pid'] + csv_headers[csv_name])

    for pid in id_dirs:
        csv_path = csv_files[csv_name].format(pid)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep='\t', header=None, 
                              names=csv_headers[csv_name])
            df = process_date_hourly(df, pid, freq) # adds pid, date columns
            daily_process_func = globals()['process_{}'.format(csv_name)]
            df_day = pd.DataFrame(daily_process_func(df, pid, **kwargs))
            df_all = df_all.append(df_day)
    df_all = df_all.dropna(axis=1, how='all')
    return df_all


def process_csv_all_raw(csv_name, **kwargs):
    """
    Aggregates all of the dataframes for the given csv name into a single dataframe, with dates untransformed.
    TODO needs to be corrected
    """
    df_all = pd.DataFrame(columns=['pid'] + csv_headers[csv_name])
    win_dict = build_emm_win_dict(window_size=window_size)
    for pid in id_dirs:
        csv_path = csv_files[csv_name].format(pid)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep='\t', header=None, 
                              names=csv_headers[csv_name])
            if pid in win_dict:
                # adds pid, date columns
                df.loc[:, 'pid'] = pd.Series([pid for x in range(len(df['timestamp']))])
                df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                
                process_func = globals()['process_{}'.format(csv_name)]
                df_window = pd.DataFrame(process_func(df, pid, **kwargs))
                df_all = df_all.append(df_window)
    df_all = df_all.dropna(axis=1, how='all')
    return df_all


def process_wtr(wtr_df, pid):
    """
    TODO needs to be implemented
    """
    wtr_df.loc[:, 'pid'] = pd.Series([pid for x in range(len(wtr_df['timestamp']))])
    wtr_df = convert_col_to_day(wtr_df, 'timestamp', 'date', unit='s')
    # TODO sunny indicator
    wtr_df_day = wtr_df.groupby(['pid', 'date'], as_index=False).mean()
    return wtr_df_day


def process_emm(emm_df, pid, normalize=False):
    """
    Processes the emm.csv, grouping and merging on 'pid' and 'date'.
    Assumes that 'pid' and 'date' columns have been populated in emm_df.
    """
    
    # emm_df.loc[:, 'pid'] = pd.Series([pid for x in range(len(emm_df['timestamp']))])
    # emm_df = convert_col_to_day(emm_df, 'timestamp', 'date', unit='s')
    emm_df = emm_df.groupby(['pid', 'date'], as_index=False).mean()
    if normalize:
        for ema in csv_headers['emm'][1:]:
            emm_df = z_score(emm_df, ema)
    return emm_df


def process_scr(scr_df, pid):
    """
    TODO verify correctness on non-daily slices
    Processes the scr.csv, grouping and merging on 'pid' and 'date'.
    Assumes that 'pid' and 'date' columns have been populated.
    """
    #modified scr_df.loc[:, 'pid'] = pd.Series([pid for x in range(len(scr_df['timestamp']))])
    
    # if the first row is a screen_state == False, drop it so that it isn't picked up
    if not scr_df.iloc[0]['screen_state']:
        scr_df = scr_df.drop(scr_df.index[0])
    
    scr_df.drop_duplicates(['timestamp', 'screen_state'], inplace=True)
    scr_df = scr_df.reset_index()

    # screen on time is recorded in the same row as the screen_state == True entry
    screen_on = []
    # TODO figure out iloc intdexing
    for idx, row in scr_df.iterrows():
        if idx < len(scr_df['timestamp'])-1:
            if scr_df.iloc[idx]['screen_state'] and (not scr_df.iloc[idx+1]['screen_state']):
                screen_on.append(scr_df.iloc[idx+1]['timestamp'] -  scr_df.iloc[idx]['timestamp'])
                continue
        screen_on.append(0)
    scr_df['screen_on'] = pd.Series(screen_on).values

    scr_df = scr_df.fillna(0)
    #modified scr_df = convert_col_to_day(scr_df, 'timestamp', 'date', unit='s')

    scr_df = scr_df.loc[scr_df['screen_state'] == True]
    
    scr_df = scr_df.groupby(['pid', 'date'], as_index=False).sum()
    scr_df = scr_df[scr_df['screen_on'] < 86400] # remove points that are more than a day long
    return scr_df[['pid', 'date', 'screen_on']]


def process_tch(tch_df, pid):
    """
    Processes the tch.csv, grouping and merging on 'pid' and 'date'.
    Assumes that 'pid' and 'date' columns have been populated.
    TODO how to process last touch delay
    """
    
    """
    modified
    tch_df.loc[:, 'pid'] = pd.Series([pid for x in range(len(tch_df['timestamp']))])
    tch_df = convert_col_to_day(tch_df, 'timestamp', 'date', unit='s')
    """

    tch_df = tch_df.groupby(['pid', 'date'], as_index=False).sum()
    tch_df['touch_count'] = tch_df['touch_count'].astype(float)
    
    return tch_df[['pid', 'date', 'touch_count']]


def process_ems(ems_df, pid, normalize=False):
    """
    Processes the ems.csv, grouping and merging on 'pid' and 'date'.
    Assumes that 'pid' and 'date' columns have been populated.
    """
    
    """
    modified
    ems_df.loc[:, 'pid'] = pd.Series([pid for x in range(len(ems_df['timestamp']))])
    ems_df = convert_col_to_day(ems_df, 'timestamp', 'date', unit='s')
    """
    
    ems_df['sleep_amount'] = ems_df['wake_time'].astype(float) - ems_df['sleep_time'].astype(float)
    if normalize:
            ems_df = z_score(ems_df, 'sleep_quality')
    ems_df['date'] = ems_df['date'] - pd.Timedelta(days=1) # subtract off the day b/c survey administered the day after
    ems_df = ems_df.groupby(['pid', 'date'], as_index=False).mean()
    #ems_df = ems_df.drop_duplicates(subset=['date']) # drop duplicate dates or average them?
    return ems_df[['pid', 'date', 'sleep_quality', 'sleep_amount']]


def process_coe(coe, pid):
    """
    Processes the coe.csv, grouping and merging on 'pid' and 'date'.
    Assumes that 'pid' and 'date' columns have been populated.
    """

    """
    modified
    coe.loc[:, 'pid'] = pd.Series([pid for x in range(len(coe['timestamp']))])
    coe = convert_col_to_day(coe, 'timestamp', 'date', unit='s')
    """
    
    trans_coe = coe.loc[(coe['comm_type'] == 'PHONE') & (coe['comm_direction'] == 'OUTGOING')].groupby(['pid', 'date'], as_index=False).count()[['pid', 'date', 'comm_direction']]
    trans_coe = trans_coe.rename(index=str, columns={'comm_direction': 'call_out_count'})

    temp = coe.loc[(coe['comm_type'] == 'PHONE') & (coe['comm_direction'] == 'INCOMING')].groupby(['pid', 'date'], as_index=False).count()[['pid', 'date', 'comm_direction']]
    temp = temp.rename(index=str, columns={'comm_direction': 'call_in_count'})
    trans_coe = pd.merge(trans_coe, temp, on=['pid', 'date'])

    temp = coe.loc[(coe['comm_type'] == 'SMS') & (coe['comm_direction'] == 'INCOMING')].groupby(['pid', 'date'], as_index=False).count()[['pid', 'date', 'comm_direction']]
    temp = temp.rename(index=str, columns={'comm_direction': 'sms_in_count'})
    trans_coe = pd.merge(trans_coe, temp, on=['pid', 'date'])

    temp = coe.loc[(coe['comm_type'] == 'SMS') & (coe['comm_direction'] == 'OUTGOING')].groupby(['pid', 'date'], as_index=False).count()[['pid', 'date', 'comm_direction']]
    temp = temp.rename(index=str, columns={'comm_direction': 'sms_out_count'})
    trans_coe = pd.merge(trans_coe, temp, on=['pid', 'date'])

    return trans_coe


def process_app(app, pid):
    """
    Processes the app.csv, grouping and merging on 'pid' and 'date'.
    Assumes that 'pid' and 'date' columns have been populated.
    """
    
    """
    modified
    app.loc[:, 'pid'] = pd.Series([pid for x in range(len(app['timestamp']))])
    app = convert_col_to_day(app, 'timestamp', 'date', unit='s')
    """

    trans_app = app.groupby(['pid', 'date'], as_index=False).count()[['pid', 'date', 'app_name']]
    trans_app = trans_app.rename(index=str, columns={'app_name': 'total_launch_count'})

    # temp = app.loc[app['app_name'] == 'Facebook'].groupby(['pid', 'date'], as_index=False).count()[['pid', 'date', 'app_name']]
    temp = app.loc[app['app_package'].str.contains('facebook', regex=False)].groupby(['pid', 'date'], as_index=False).count()[['pid', 'date', 'app_name']]
    temp = temp.rename(index=str, columns={'app_name': 'facebook_launch_count'})
    trans_app = pd.merge(trans_app, temp, on=['pid', 'date'], how='outer')

    temp = app.loc[app['app_category'] == 'Social'].groupby(['pid', 'date'], as_index=False).count()[['pid', 'date', 'app_name']]
    temp = temp.rename(index=str, columns={'app_name': 'social_launch_count'})
    trans_app = pd.merge(trans_app, temp, on=['pid', 'date'], how='outer')

    temp = app.loc[app['app_category'] == 'Communication'].groupby(['pid', 'date'], as_index=False).count()[['pid', 'date', 'app_name']]
    temp = temp.rename(index=str, columns={'app_name': 'comm_launch_count'})
    trans_app = pd.merge(trans_app, temp, on=['pid', 'date'], how='outer')

    trans_app = trans_app.fillna(0)

    for col in trans_app.columns.values:
        if col != 'pid' and col != 'date':
            trans_app[col] = trans_app[col].astype(float)
            
    return trans_app

def process_run(run, pid, package_str, use_regex=False):
    """
    Processes the run.csv, grouping and merging on 'pid' and 'date'.
    Assumes that 'pid' and 'date' columns have been populated.
    """
    
    """
    modified
    run.loc[:, 'pid'] = pd.Series([pid for x in range(len(run['timestamp']))])
    run = convert_col_to_day(run, 'timestamp', 'date', unit='s')
    """
    
    all_stack_str = package_str + '_all_stack_count'
    top_stack_str = package_str + '_top_stack_count'
    
    top_stack_count = run.loc[run['task_stack_index'] <= 1] 
    top_stack_count = top_stack_count.loc[top_stack_count['app_package'].str.contains(package_str, regex=use_regex)]
    top_stack_count = top_stack_count.groupby(['pid', 'date'], as_index=False).count()[['pid', 'date', 'app_category']]
    top_stack_count = top_stack_count.rename(index=str, columns={'app_category': top_stack_str})
    
    all_stack_count = run.loc[run['app_package'].str.contains(package_str, regex=use_regex)]
    all_stack_count = all_stack_count.groupby(['pid', 'date'], as_index=False).count()[['pid', 'date', 'app_category']]
    all_stack_count = all_stack_count.rename(index=str, columns={'app_category': all_stack_str})
    
    top_stack_count = pd.merge(top_stack_count, all_stack_count, on=['pid', 'date'])
    return top_stack_count[['pid', 'date', all_stack_str, top_stack_str]]


def process_act(act, pid, confidence=50):
    """
    Processes the act.csv, grouping and merging on 'pid' and 'date.'
    Assumes that 'pid' and 'date' columns have been populated.

    filters out movements that are below the given confidence threshold.
    """

    act = act[act['confidence'] >= confidence]
    act = act.groupby(['pid', 'date', 'activity_type'], as_index=False).size().unstack(fill_value=0)
    act = act.reset_index()

    return act


def process_fus(fus, cluster_radius=0.2):
    """Processes the fus.csv, grouping and merging on 'pid' and 'date.'

    Args:
        cluster_radius (float): the stationary cluster radius, in km
    Assumes date, pid columns are populated
    """
    
    # get stationary locations
    fus['prev_lat'] = fus['latitude'].shift()
    fus['prev_long'] = fus['longitude'].shift()
    fus['dist'] = fus.apply(lambda x: haversine((x.latitude, x.longitude), (x.prev_lat, x.prev_long)), axis=1) # in km
    fus['prev_timestamp'] = fus['timestamp'].shift()
    fus['delta_timestamp'] = ((fus['timestamp'] - fus['prev_timestamp']).dt.total_seconds() / (60 * 60)).astype(float) # change to hours
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
    
    return fus_combined
