"""
LifeSense feature creation pipeline.

Moves sensor processing present in wk_feature_processing into functions

Moves semantic location processing in lifesense_cluster_processing into functions
"""

import argparse
import numpy as np
import pandas as pd
import pickle
from multiprocessing import Pool

# TODO refactor lifesense_utils into a feature extract file in data_processing
# https://stackoverflow.com/questions/49022482/python-3-5-cannot-import-a-module
#from lifesense.utils.lifesense_utils import *
import sys
sys.path.insert(1, '/home/tliu/lifesense/utils')
import lifesense_utils as lsu

#%% functions
def process_sensor_data(pids, loc, out_loc, func, n_procs=4):
    """Wrapper function for processing sensor data.
    
    Args:
        pids (list): list of pids to process
        #wk (int): the week of data to process
        loc (str): the file location
        out_loc (str): the output file name and location
        func (function): the processing function to apply
        n_procs (int): the number of processes to spin up
    
    Returns:
        None, but writes to out_loc
    """
    func_args = [(pid, loc) for pid in pids]
    with Pool(n_procs) as pool:
        results = pool.starmap(func, func_args)
        
    df = pd.DataFrame()

    for res in results:
        df = df.append(res)
    
    df.to_pickle(out_loc)


def process_sloc(pids, sloc_df, in_loc, out_loc, n_procs=4):
    """
    Builds and dumps raw semantic location df
    
    Args:
        pid (str): participant id
        sloc_df (df): the semantic location DataFrame loaded from file
        in_loc (str): the file location for the location df
        out_loc (str): the file location for the target location df
        n_procs (int): the number of processes to spin up
    
    Returns:
        None, but writes to out_loc/semantic-location/{pid}.df
    """
    func_args = [(pid, sloc_df, in_loc, out_loc) for pid in pids]

    with Pool(n_procs) as pool:
        pool.starmap(lsu.build_sloc, func_args)


def build_circ_window(pid, in_loc, start_date, end_date):
    """Builds circadean movement features within the given window.
    
    TODO refactor with build_circadean_stats
    
    Assumes that the start_date and end_date fall within the given week.
    
    
    Args:
        - pid (str): the participant id
        - in_loc (str): the file path to the input fus location
        - fus_df (pd.DataFrame): a fus_df frame with properly formatted date columns
        - start_date (str): start date, yyyy-mm-dd
        - end_date (str): end date, yyyy-mm-dd
        
    Returns:
        - circ_df (pd.DataFrame): frame with circadean movement statistics calculated for the given time window
    """
    print(pid)
    loc = "{}/{}.df".format(in_loc, pid)
    fus_df = pd.read_pickle(loc)
    if fus_df.shape[0] < 1:
        return
    
    fus_df = lsu.format_raw_fus(fus_df)
    fus_df = fus_df.sort_values(by='timestamp')
    fus_df = fus_df[(fus_df['date'] >= start_date) & (fus_df['date'] <= end_date)]
    print(fus_df.shape)
    fus_df["is_wkday"] = (pd.to_datetime(fus_df['date']).dt.dayofweek < 5).astype(float)

    circ_df = pd.DataFrame()
    circ_df['pid'] = [pid]
    circ_df['circ_movt_tot'] = lsu.get_circadian_movement(fus_df) 
    circ_df['circ_movt_wkday'] = lsu.get_circadian_movement(fus_df.loc[fus_df['is_wkday'] == 1]) 
    circ_df['circ_movt_wkend'] = lsu.get_circadian_movement(fus_df.loc[fus_df['is_wkday'] == 0]) 
    
    return circ_df


def build_circ_dict(seq_df, in_loc, target, pre_days=3, post_days=3):
    """Builds a (col, [vals]) dictionary for circ features.
    
    Has to be processed separately because circadean movement cannot be aggregated to daily values.

    Args:
        seq_df (pd.df): DataFrame holding survey dates
        in_loc (str): the file location for the location df
        target (str): the target date column
        pre_days (int): the number of days to 'look back' from survey
        post_days (int): the number of days to 'look forward' from survey

    Returns:
        circ_dict:  (pid, columns) k,v pairs
    """
    col_dict = {}
    
    cols = ['circ_movt_tot', 'circ_movt_wkday', 'circ_movt_wkend']
    for col in cols:
        col_dict[col] = []


    seq_df['date'] = seq_df[target].dt.floor('D')
    for idx, row in seq_df.iterrows():
        if pd.isna(row[target]):
            for col in cols:
                col_dict[col].append(np.nan)
            continue
        else:
            date = row['date']
            pid = row['pid']
            wk = row['study_wk']
            sel_df = build_circ_window(pid, in_loc, (date.floor('D') - pd.Timedelta(pre_days, unit='D')),                                      
                                                (date.floor('D') + pd.Timedelta(post_days, unit='D')))
            for col in cols:
                if sel_df is not None:
                    col_dict[col].append(sel_df[col])
                else:
                    col_dict[col].append(np.nan)
                
    return col_dict


def build_fus_helper(pid, wk, in_loc, start_date, end_date):
    """Builds fused location features within the given window.

    Assumes that the start_date and end_date fall within the given week.
    
    Args:
        - pid (str): the participant id
        - wk (int): the study week
        - in_loc (str): the file path to the input fus location
        - start_date (str): start date, yyyy-mm-dd
        - end_date (str): end date, yyyy-mm-dd
        
    Returns:
        - stats_df (pd.DataFrame): frame with fused location statistics calculated for the given time window
    """
    
    loc = "{}/{}.df".format(in_loc, pid)
    fus_df = pd.read_pickle(loc)

    if fus_df.shape[0] < 1:
        return
    
    fus_df = lsu.format_raw_fus(fus_df)
    fus_df = fus_df[(fus_df['date'] >= start_date) & (fus_df['date'] <= end_date)]
    print(fus_df.shape)
    fus_df["is_wkday"] = (pd.to_datetime(fus_df['date']).dt.dayofweek < 5).astype(float)

    wkend_stats = None
    wkday_stats = None
    total_stats = None

    sel_df = fus_df[fus_df["is_wkday"] == 0].copy()
    if sel_df.shape[0] > 0:
        wkend_stats = lsu.process_fus_daily(sel_df)

    sel_df = fus_df[fus_df["is_wkday"] == 1].copy()
    if sel_df.shape[0] > 0:
        wkday_stats = lsu.process_fus_daily(sel_df)

    sel_df = fus_df.copy()
    if sel_df.shape[0] > 0:
        total_stats = lsu.process_fus_daily(sel_df)
    
    dfs = []
    
    if wkend_stats is not None:
        wkend_stats = wkend_stats.set_index('pid')
        wkend_stats = wkend_stats.add_suffix("_wkend")
        dfs.append(wkend_stats.mean())
    
    if wkday_stats is not None:
        wkday_stats = wkday_stats.set_index('pid')
        wkday_stats = wkday_stats.add_suffix("_wkday")
        dfs.append(wkday_stats.mean())
    
    if total_stats is not None:
        total_stats = total_stats.set_index('pid')
        total_stats = total_stats.add_suffix("_total")
        dfs.append(total_stats.mean())
    
    stats_df = pd.DataFrame()
    if len(dfs) > 0:
        stats_df = pd.concat(dfs)
        stats_df = stats_df.to_frame().transpose()
        stats_df['pid'] = pid
        stats_df['study_wk'] = wk

    # TODO figure out why we're returning complex numbers
    stats_df[stats_df.select_dtypes('complex128').columns] = stats_df.select_dtypes('complex128').astype(float)

    return stats_df
    

def process_fus(seq_df, target, in_loc, pre_days, post_days, out_loc, n_procs=4):
    """
    Builds and dumps raw semantic location df
    
    Args:
        seq_df (pd.df): DataFrame holding survey dates
        target (str): the target date column
        in_loc (str): the file location for the location df
        out_loc (str): the file location for the target location df
        pre_days (int): the number of days to 'look back' from survey
        post_days (int): the number of days to 'look forward' from survey
        n_procs (int): the number of processes to spin up

    Returns:       
        None, but writes to {out_loc}/fus_{target}_{pre_days}_{post_days}.df
    """
    func_args = []

    seq_df['date'] = seq_df[target].dt.floor('D')
    for _, row in seq_df.iterrows():
        date = row['date']
        pid = row['pid']
        wk = row['study_wk']
        
        start_date = date.floor('D') - pd.Timedelta(pre_days, unit='D')    
        end_date = date.floor('D') + pd.Timedelta(post_days, unit='D')
    
        func_args.append((pid, wk, in_loc, start_date, end_date))

    with Pool(n_procs) as pool:
        results = pool.starmap(build_fus_helper, func_args)

    df = pd.DataFrame()

    for res in results:
        df = df.append(res)
    
    df.to_pickle(out_loc)

if __name__ == '__main__':
    script_description = "Script for processing raw lifesense data into features"
    parser = argparse.ArgumentParser(description=script_description)
    
    parser.add_argument('in_loc', type=str,
                        help='input location')
    parser.add_argument('out_loc', type=str, 
                        help='output location')
    parser.add_argument('id_file', type=str, 
                        help="file containing participant ids, newline delimited")
    parser.add_argument('n_procs', type=int, default=2, 
                        help="the number of processes to allocate (default 2)")
    # simple aggregation functions
    parser.add_argument('--cal', action='store_true', 
                        help="process phone features")
    parser.add_argument('--fga', action='store_true', 
                        help="process foreground application features")
    parser.add_argument('--sms', action='store_true', 
                        help="process text message features")
    # location feature generation
    parser.add_argument('--sloc_raw', action='store_true', 
                        help="process raw semantic location features (requires --sloc_df)")
    parser.add_argument('--sloc_df', help="sloc_df DataFrame location")
    parser.add_argument('--sloc', action='store_true', 
                        help="process semantic location features (requires raw sloc to be processed)")
    parser.add_argument('--fus', action='store_true', 
                        help="process fused location features (requires seq_df, target, pre/post days)")
    parser.add_argument('--circ', action='store_true', 
                        help="process circadean rhythm features (requires seq_df, target, pre/post days)")
    parser.add_argument('--seq_df', help='seq_df DataFrame location')
    parser.add_argument('--pre_days', type=int, help="number of days to 'look back'")
    parser.add_argument('--post_days', type=int, help="number of days to 'look forward'")
    parser.add_argument('--target', type=str, help="target date column to shift by")

    args = parser.parse_args()

    fus_in = "{}/pdk-location/"
    fga_in = "{}/pdk-foreground-application/"
    cal_in = "{}/pdk-phone-calls/"
    sms_in = "{}/pdk-text-messages/"
    sloc_in = "{}/semantic-location/"

    fus_out = "{}/fus_{}_{}_{}.df"
    circ_out = "{}/circ_movt.df"
    fga_out = "{}/fga_hr.df"
    cal_out = "{}/cal_hr.df"
    sms_out = "{}/sms_hr.df"
    sloc_raw = "{}/semantic-location/"
    sloc_out = "{}/sloc_hr.df"
    circ_out = "{}/circ_{}_{}_{}.dict"

    pids = []
    with open(args.id_file, "r") as wave_f:
        for line in wave_f.readlines():
            pids.append(line.strip())
    
    process_args = []
    if args.cal:
        process_args.append((pids, 
                             cal_in.format(args.in_loc), 
                             cal_out.format(args.out_loc),
                             lsu.build_cal_hr,
                             args.n_procs))
    if args.sms:
        process_args.append((pids, 
                             sms_in.format(args.in_loc), 
                             sms_out.format(args.out_loc),
                             lsu.build_sms_hr,
                             args.n_procs))
    if args.fga:
        process_args.append((pids,
                             fga_in.format(args.in_loc), 
                             fga_out.format(args.out_loc),
                             lsu.build_fga_hr,
                             args.n_procs))
    if args.sloc_raw:
        print(args.sloc_df)
        # process and dump semantic location "raw" files
        
        # TODO assumes sloc DataFrame is already generated
        # source: lifesense_cluster_processing.ipynb
        semantic_locs = pd.read_pickle(args.sloc_df)

        process_sloc(pids, 
                     semantic_locs, 
                     fus_in.format(args.in_loc),
                     sloc_raw.format(args.out_loc),
                     args.n_procs)
    if args.sloc:
        process_args.append((pids, 
                        sloc_in.format(args.in_loc), 
                        sloc_out.format(args.out_loc),
                        lsu.build_sloc_hr,
                        args.n_procs))

    if args.circ:
        # TODO assumes seq DataFrame is already generated
        # source: location_aggregation.ipynb
        seq_df = pd.read_pickle(args.seq_df)    
        circ_dict = build_circ_dict(seq_df,
                                    fus_in.format(args.in_loc),
                                    args.target,
                                    args.pre_days,
                                    args.post_days)

        out_loc = circ_out.format(args.out_loc, 
                                  args.target,
                                  args.pre_days,
                                  args.post_days)
        pickle.dump(circ_dict, open(out_loc, 'wb'), -1)

    if args.fus:
        seq_df = pd.read_pickle(args.seq_df)

        out_loc = fus_out.format(args.out_loc, 
                                  args.target,
                                  args.pre_days,
                                  args.post_days)

        process_fus(seq_df, args.target, 
                    fus_in.format(args.in_loc),
                    args.pre_days, 
                    args.post_days, 
                    out_loc, args.n_procs)


    for tup in process_args:
        process_sensor_data(*tup)
    
    
    
