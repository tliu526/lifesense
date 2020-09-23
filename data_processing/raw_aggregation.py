"""
Script for aggregating the raw data. Step 1 for incremental updates of phone sensor data
"""
#%% imports
import argparse
import multiprocessing
import os
import pandas as pd
import pickle

#%% merge_all function
def merge_all(pid, generator, in_loc, debug=False):
    """Merges all files for the specified pid and generator at the specified in_loc.

    Merges over all sub-directories found in the in_loc directory.

    Args:
        pid (str): participant id
        generator (str): sensor generator to target
        in_loc (str): the file directory to pull data from
    Returns:
        pd.df: aggregated DataFrame for everything in the given folder location
    """
    combined_df = pd.DataFrame()
    subdirs = sorted(os.listdir(in_loc))

    gen_path = in_loc + "/{}/{}/{}.df"

    for subdir in subdirs:        
        file_loc = gen_path.format(subdir, generator, pid)
        if debug: print(file_loc)
        if os.path.isfile(file_loc):
            df = pd.read_pickle(file_loc)
            if debug: print(df.shape)
            if df.shape[0] > 0:
                combined_df = combined_df.append(df)

    combined_df = combined_df.reset_index(drop=True)
    if debug: print("Final size:" + str(combined_df.shape))
    return combined_df


def aggregate_sensors(pid, sensors, in_loc, out_loc, debug=False):
    print(pid)
    for generator in sensors:
        df = merge_all(pid, generator, in_loc, debug)
        write_path = out_loc + "/{}/{}.df".format(generator, pid)
        df.to_pickle(write_path)


if __name__ == '__main__':
    script_description = "Aggregates incremental data found in in_loc and dumps to out_loc"
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('in_loc', type=str,
                        help='input location')
    parser.add_argument('out_loc', type=str, 
                        help='output location')
    parser.add_argument('id_file', type=str, 
                        help="file containing participant ids, newline delimited")
    parser.add_argument('sensor_file', type=str, 
                        help="a file containing sensors to pull, separated by newlines")
    parser.add_argument('n_procs', type=int, default=2, 
                        help="the number of processes to allocate (default 2)")
    
    # TODO integrate these eventually?
    #parser.add_argument('start_date', type=str, help="the start date (inclusive) for parsing data (yyyy-mm-dd format)")
    #parser.add_argument('end_date', type=str, help="the end date (exclusive) for parsing data (yyyy-mm-dd format)")
    parser.add_argument('--debug', action='store_true', help="Whether to run in debug mode")

    args = parser.parse_args()

    pids = []
    with open(args.id_file, "r") as wave_f:
        for line in wave_f.readlines():
            pids.append(line.strip())

    sensors = []
    with open(args.sensor_file, "r") as sensor_f:
        for line in sensor_f.readlines():
            sensors.append(line.strip())

    f_args = [(pid, sensors, args.in_loc, args.out_loc, args.debug) for pid in pids]

    with multiprocessing.Pool(args.n_procs) as pool:
        pool.starmap(aggregate_sensors, f_args)



