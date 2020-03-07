"""
Utilities for processing lifesense data files.
"""

import pandas as pd
import numpy as np
from haversine import haversine
from scipy.signal import lombscargle
from sklearn.cluster import KMeans
from geopy.distance import distance

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
    """Processes daily fused location features.
    
    Assumes date, pid columns are populated.

    This includes average velocity, number of stationary clusters, location entropy, and location variance.

    For reference, see https://github.com/sosata/MobileDepression/tree/master/features and 
    Saeb et al: The relationship between mobile phone location sensor data and depressive symptom severity

    Args:
        fus (pd.DataFrame): the fused location df with latitude, longitude, date, and pid columns

    """
    
    if fus.shape[0]  < 1:
        return None
    
    # get stationary locations
    fus['prev_lat'] = fus['latitude'].shift()
    fus['prev_long'] = fus['longitude'].shift()
    fus['dist'] = fus.apply(lambda x: haversine((x.latitude, x.longitude), (x.prev_lat, x.prev_long)), axis=1) # in km
    fus['prev_timestamp'] = fus['timestamp'].shift()
    fus['delta_timestamp'] = ((fus['timestamp'] - fus['prev_timestamp']) / (60 * 60)).astype(float) # change to hours
    fus['velocity'] = fus['dist'] / fus['delta_timestamp']
    fus = fus[fus['velocity'] >= 0] # drop rows with negative velocities
    fus['stationary'] = fus['velocity'] < 1 # filter locations that have a speed of greater than 1 km/hr
    fus_stationary = fus[fus['stationary']]
    
    #return fus, fus_stationary

    if fus_stationary.shape[0] < 1:
        return None
        
    # TODO should this be stationary or all locations?
    loc_var = np.log(fus_stationary['latitude'].var() + fus_stationary['longitude'].var())

    # assign clusters
    cur_max = 1
    cur_clusters = 0
    while cur_max > cluster_radius:
        cur_clusters += 1
        X = fus_stationary[['latitude', 'longitude']]
        kmeans = KMeans(n_clusters=cur_clusters, random_state=0).fit(X)
        labels = kmeans.labels_
        clusters = kmeans.cluster_centers_
        X = X.reset_index(drop=True)
        X['labels'] = labels
        X['center'] = X.apply(lambda x: clusters[int(x.labels)], axis=1)
        X['dist'] = X.apply(lambda x: haversine((x.latitude, x.longitude), x.center), axis=1)
        cur_max = X['dist'].max()
    
    # get daily entropy
    fus_stationary = fus_stationary.reset_index(drop=True)
    fus_stationary['cluster'] = X['labels']
    label_group = fus_stationary.groupby(['date', 'cluster'])['delta_timestamp'].sum().unstack()
    label_group = label_group.fillna(0)
    label_group['total'] = label_group.sum(axis=1)
    label_group = label_group.div(label_group['total'], axis=0)
    label_group['entropy'] = -(np.log(label_group) * label_group).sum(axis=1)
    label_group['norm_entropy'] = label_group['entropy'] / np.log(cur_clusters) # entropy normalized by number of location clusters
    label_group['norm_entropy'] = label_group['norm_entropy'].fillna(0) # fill divide by 0 with 0, as this corresponds to no location variance
    label_group = label_group.reset_index()
    
    fus_combined = fus.groupby(['pid', 'date'], as_index=False)['dist'].sum()
    fus_combined = pd.merge(fus_combined, label_group[['date', 'entropy', 'norm_entropy']], on='date', how='outer')
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
    
    Basing implementation off of 
    https://github.com/sosata/CS120DataAnalysis/blob/master/features/estimate_circadian_movement.m
    
    """
    # frequency range of 24 +- 0.5 hrs at a minute granularity
    freq = np.linspace(86400-30*60, 86400+30*60, 60)
    # scipy implementation requires angular frequency
    freq = 2 * np.pi / freq 
    try:
        # make sure to precenter each time series
        energy_lat = sum(lombscargle(fus_df['timestamp'], fus_df['latitude'], freq, precenter=True, normalize=True))
        energy_long = sum(lombscargle(fus_df['timestamp'], fus_df['longitude'], freq, precenter=True, normalize=True))
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


def build_circ_window(pid, wk, start_date, end_date):
    """Builds circadean movement features within the given window.
    
    TODO refactor with build_circadean_stats
    
    Assumes that the start_date and end_date fall within the given week.
    
    
    Args:
        - pid (str): the participant id
        - wk (int): the study week to pull data from: 1, 4, 7, 10, 13, 16
        - fus_df (pd.DataFrame): a fus_df frame with properly formatted date columns
        - start_date (str): start date, yyyy-mm-dd
        - end_date (str): end date, yyyy-mm-dd
        
    Returns:
        - circ_df (pd.DataFrame): frame with circadean movement statistics calculated for the given time window
    """
    print(pid)
    loc = "/data/tliu/wk{}_ls_data/pdk-location/{}.df".format(wk, pid)
    fus_df = pd.read_pickle(loc)
    if fus_df.shape[0] < 1:
        return
    
    fus_df = format_raw_fus(fus_df)
    fus_df = fus_df[(fus_df['date'] >= start_date) & (fus_df['date'] <= end_date)]
    print(fus_df.shape)
    fus_df["is_wkday"] = (pd.to_datetime(fus_df['date']).dt.dayofweek < 5).astype(float)

    circ_df = pd.DataFrame()
    circ_df['pid'] = [pid]
    circ_df['circ_movt_tot'] = get_circadian_movement(fus_df) 
    circ_df['circ_movt_wkday'] = get_circadian_movement(fus_df.loc[fus_df['is_wkday'] == 1]) 
    circ_df['circ_movt_wkend'] = get_circadian_movement(fus_df.loc[fus_df['is_wkday'] == 0]) 
    
    return circ_df


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
    """Builds foreground application aggregation DataFrame"""

    fga_df = pd.read_pickle("{}/{}.df".format(loc, pid))
    if fga_df.shape[0] < 1:
        return 
    fga_df = format_time(fga_df)
    fga_hr = pd.DataFrame()
    fga_slim = fga_df[['hour', 'time', 'screen_active', 'application', 'adj_ts']]

    # need to drop NaN applications else groupby frames won't work
    fga_slim = fga_slim.dropna(subset=["application"])

    for time, group in fga_slim.groupby("hour"):
        hr = pd.DataFrame(process_fga_time(time, group), index=[0])
        fga_hr = fga_hr.append(hr)

    fga_hr = fga_hr.set_index('hr')
    fga_hr = fga_hr.resample('1H').sum()
    fga_hr = fga_hr.reset_index()
    fga_hr['pid'] = pid

    return fga_hr



def tag_semantic_locs(pid, sloc_df, file_loc, cluster_rad=500):
    """
    Tags each location sensor reading with a semantic label, if applicable.
    
    We only use labelled locations from the same week of data collection, or earlier.
    
    Args:
        pid (str): participant id
        sloc_df (df): the semantic location DataFrame loaded from file
        file_loc (str): the file location for the location df
        cluster_rad (int): the maximum cluster radius
        
    Returns:
        "raw" DataFrame with long/lat labelled
    """
    print(pid)
    loc_df = pd.read_pickle("{}/{}.df".format(file_loc, pid))
    if loc_df.shape[0] < 1:
        return 
    loc_df = format_time(loc_df)
    sloc_pid = sloc_df.loc[sloc_df['pid'] == pid]
    sloc_pid = sloc_pid[sloc_pid['date'] <= max(loc_df['date'])]
    places = []

    for i, loc_row in loc_df.iterrows():

        dist = cluster_rad + 5
        for j, sloc_row in sloc_pid.iterrows():
            dist = distance((loc_row['latitude'], loc_row['longitude']), (sloc_row['place-latitude'], sloc_row['place-longitude'])).m
            if dist < cluster_rad:
                break
                
        if dist < cluster_rad:
            places.append(sloc_row['place-kind'])
        else:
            places.append(np.nan)

    loc_df['place-kind'] = places
    
    return loc_df[['pid', 'date', 'time', 'latitude', 'longitude', 'place-kind']]
    

def build_sloc(pid, sloc_df, in_loc, out_loc):
    """Builds and dumps raw semantic location df
    """    
    df = tag_semantic_locs(pid, sloc_df, in_loc)
    pd.to_pickle(df, "{}/{}.df".format(out_loc, pid))



sloc_map = {
    "Food and Drink" : "food",
    "Home" : "home",
    "Work" : "work",
    "Gym/Exercise" : "exercise",
    "Another Person's Home" : "anothers_home",
    "Place of Worship (Church, Temple, Etc.)" : "religion",
    "Commute/Travel (Airport, Bus Stop, Train Station, Etc.)" : "travel",
    "Shopping" : "shopping",
    "Errand" : "errand",
    "Medical/Dentist/Mental Health" : "health",
    "Education" : "education",
    "Entertainment" : "entertainment",
    "Other..." : "other",
    np.nan : "n/a"
}


def process_transition_hr(time, sloc_group):
    """Helper for building sloc feature DataFrame."""
    num_transitions = 0
    transition_dict = {}
    transition_dict['hr'] = time
    
    for sloc in sloc_map.values():
        transition_dict[sloc + '_dur'] = 0

    for sloc_i in sloc_map.values():
        for sloc_j in sloc_map.values():
            if sloc_i is not sloc_j:
                transition_dict[sloc_i + '_' + sloc_j] = 0
    
    cur_loc = sloc_group.iloc[0]['place-kind-fmt']
    cur_time = sloc_group.iloc[0]['time']
    for i, row in sloc_group.iterrows():
        next_loc = row['place-kind-fmt']
        next_time = row['time']
        if next_loc is not cur_loc:
            num_transitions += 1
            transition_dict[cur_loc + '_dur'] += (next_time - cur_time).total_seconds()
            transition_dict[cur_loc + '_' + next_loc] += 1
            cur_loc = next_loc
            cur_time = next_time
    
    # at the bottom of the hour
    transition_dict[cur_loc + '_dur'] += ((time + pd.Timedelta(1, unit='h')) - cur_time).total_seconds()
    
    transition_dict['tot_tansitions'] = num_transitions
    #print(transition_dict)
    return transition_dict


def build_sloc_hr(pid, loc):
    """Builds semantic location aggregation DataFrame"""
    print(pid)
    sloc_hr = pd.DataFrame()
    sloc_pid = pd.read_pickle("{}/{}.df".format(loc, pid))
    if sloc_pid is None:
        return
    if sloc_pid.shape[0] < 1:
        return
    
    sloc_pid['hour'] = sloc_pid['time'].dt.floor('H')
    sloc_pid['place-kind-fmt'] = sloc_pid['place-kind'].map(sloc_map)
    sloc_pid['place-kind-fmt'] = sloc_pid['place-kind-fmt'].fillna('other')

    for time, group in sloc_pid.groupby("hour"):
        sl = pd.DataFrame(process_transition_hr(time, group), index=[0])
        sloc_hr = sloc_hr.append(sl)

    sloc_hr = sloc_hr.set_index('hr')
    sloc_hr = sloc_hr.resample('1H').sum()
    sloc_hr = sloc_hr.reset_index()
    sloc_hr['pid'] = pid

    return sloc_hr
    