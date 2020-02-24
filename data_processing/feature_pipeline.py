"""
LifeSense feature creation pipeline.

Moves sensor processing present in wk_feature_processing into functions
"""

#%% imports
import argparse
import pandas as pd
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
    
    parser.add_argument('--cal', action='store_true', 
                        help="process phone features")
    parser.add_argument('--fga', action='store_true', 
                        help="process foreground application features")
    parser.add_argument('--sms', action='store_true', 
                        help="process text message features")
    # TODO
    parser.add_argument('--fus', action='store_true', 
                        help="TODO process fused location features")
    parser.add_argument('--sloc', action='store_true', 
                        help="TODO process semantic location features")

    args = parser.parse_args()

    fus_in = "{}/pdk-location/"
    fga_in = "{}/pdk-foreground-application/"
    cal_in = "{}/pdk-phone-calls/"
    sms_in = "{}/pdk-text-messages/"

    fus_out = "{}/fus_daily.df"
    circ_out = "{}/circ_movt.df"
    fga_out = "{}/fga_hr.df"
    cal_out = "{}/cal_hr.df"
    sms_out = "{}/sms_hr.df"

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

    for tup in process_args:
        process_sensor_data(*tup)
    
    
    