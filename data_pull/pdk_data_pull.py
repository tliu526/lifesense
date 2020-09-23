"""
Offline, threaded script for PDK data pull.

Requires python 2.7, for the pdk_client module

"""

import argparse
import json
import pickle
import signal
import time

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

from requests.exceptions import HTTPError

from pdk_client import PDKClient
from multiprocessing import Pool
from functools import partial

# client setup
SITE_URL = "https://lifesense.fsm.northwestern.edu/data"
TOKEN = "oCZUlrEtmlJRsy9b8fOpNTthlMPT7kU6GcniFLt0yLH4Yz0ExkGeflPuaPlOwCyj"
PAGE_SIZE = 100
client = PDKClient(site_url=SITE_URL, token=TOKEN)


def try_query(pid):
    """Tries, and retries queries"""
    retries = 1
    while True:
        try:
            query = client.query_data_points(page_size=PAGE_SIZE, source=pid)
            return query
        except HTTPError as e:
            if retries > 10:
                raise e
            print(e)
            wait = retries * 15
            time.sleep(wait)
            retries += 1



def try_filter(query, pid, gen_id, start_date, end_date):
    """Tries, and retries query filters"""
    retries = 1
    while True:
        try:
            q_filter = query.filter(source=pid, 
                         generator_identifier=gen_id,
                         created__gte=start_date,
                         created__lte=end_date).order_by('created')
            return q_filter
        except HTTPError as e:
            if retries > 10:
                raise e
            print(e)
            wait = retries * 15
            print("Sleeping for {}".format(wait))
            time.sleep(wait)
            retries += 1

# Processing functions
def process_generators(pid, generators, data_source, out_loc, start_date, end_date, debug=False):
    """Processes the data for the given generators and id.

    Dumps the resulting DataFrames into pickled files.

    Args:
        pid (str): id to query
        generators (list): list of generators to pull
        data_source (str): indicates which group the ids belong to
        out_loc (str): the output location
        start_date (str): the start date of the filter, in yyyy-mm-dd form
        end_date (str): the end date of the filter, in yyyy-mm-dd form
        debug (bool): optionally print debugging information

    """
    print(pid)
    #query = client.query_data_points(page_size=PAGE_SIZE, source=pid)
    query = try_query(pid)
    for gen_id in generators:
        print(gen_id)
        if gen_id == 'pdk-location':
            process_location(pid, data_source, out_loc, start_date, end_date)
            continue

        """ ema_query = query.filter(source=pid, 
                                 generator_identifier=gen_id,
                                 created__gte=start_date,
                                 created__lte=end_date).order_by('created') """
        ema_query = try_filter(query, pid, gen_id, start_date, end_date)
        tot_count = ema_query.count()
        count = 0
        frac = int(tot_count / 100)
        ema_df = pd.DataFrame()
        for point in ema_query:
            point_df = json_normalize(point)   
            point_df.columns = point_df.columns.str.replace("passive-data-metadata.", "", regex=False)
            
            """
            point_df = pd.DataFrame.from_dict(point).iloc[0].to_frame().transpose()
            metadata_df = pd.Series(point['passive-data-metadata']).to_frame().transpose()

            point_df.reset_index(inplace=True, drop=True)
            point_df = pd.concat([metadata_df, point_df], axis=1, sort=True)
            
            point_df.drop('passive-data-metadata', axis='columns', inplace=True)
            """

            ema_df = ema_df.append(point_df)
            count += 1
            if debug and (count % frac == 0):
                print("{0:.2f}% complete".format(float(count)/float(tot_count)*100))

        ema_df['pid'] = pid 
        ema_df['data_source'] = data_source
        ema_df = ema_df.reset_index(drop=True)
        print(ema_df.shape)
        #display(ema_df.head())
        pickle.dump(ema_df, open("{}/{}/{}.df".format(out_loc, gen_id, pid), 'wb'), -1)


def process_location(pid, data_source, out_loc, start_date, end_date, debug=False):
    """Processes location data for the given id and date range.

    Dumps the resulting DataFrames into pickled files.

    Args:
        pid (str): id to query
        data_source (str): indicates which group the ids belong to
        out_loc (str): the output location
        
    """

    #query = client.query_data_points(page_size=PAGE_SIZE, source=pid)
    query = try_query(pid)

    location_query = try_filter(query, pid, 'pdk-location', start_date, end_date)
    """ location_query = query.filter(source=pid, 
                                  generator_identifier='pdk-location',
                                  created__gte=start_date,
                                  created__lte=end_date).order_by('created')
    """
    tot_count = location_query.count()
    count = 0
    frac = int(tot_count / 100)

    loc_df = pd.DataFrame()
    for point in location_query:
        point_df = pd.DataFrame.from_dict(point).iloc[0].to_frame().transpose()
        metadata_df = pd.Series(point['passive-data-metadata']).to_frame().transpose()
        # TODO check if ignoring errors is safe
        metadata_df = metadata_df.drop(['latitude', 'longitude'], axis='columns', errors="ignore")
        point_df.reset_index(inplace=True, drop=True)
        point_df = pd.concat([metadata_df, point_df], axis=1, sort=True)
        
        point_df.drop('passive-data-metadata', axis='columns', inplace=True)
        missing_cols = [col for col in loc_df.columns.values if col not in point_df.columns.values]
        
        if len(missing_cols) > 0 and loc_df.shape[0] > 0:
            for col in missing_cols:
                point_df[col] = np.nan
            point_df = point_df[loc_df.columns]
        loc_df = loc_df.append(point_df)
        count += 1
        if debug and (count % frac == 0):
            print("{0:.2f}% complete".format(float(count)/float(tot_count)*100))

    loc_df['pid'] = pid 
    loc_df['data_source'] = data_source
    print(loc_df.shape)
    
    pickle.dump(loc_df, open("{}/pdk-location/{}.df".format(out_loc, pid), 'wb'), -1)


if __name__ == '__main__':
    script_description = "Process PDK sensors and dumps them to a Pandas dataframe."
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('id_file', type=str, help="a file containing participant ids, separated by newlines")
    parser.add_argument('sensor_file', type=str, help="a file containing sensors to pull, separated by newlines")
    parser.add_argument('data_source', type=str, help="the data source name (e.g. wave1, wave2)")
    parser.add_argument('start_date', type=str, help="the start date (inclusive) for parsing data (yyyy-mm-dd format)")
    parser.add_argument('end_date', type=str, help="the end date (exclusive) for parsing data (yyyy-mm-dd format)")

    parser.add_argument('out_loc', type=str, help="the output directory location")
    parser.add_argument('num_procs', type=int, default=2, help="the number of processes to allocate (default 2)")
    parser.add_argument('--debug', action='store_true', help="Whether to run in debug mode")
    # TODO parameterize
    #parser.add_argument('pdk_token', type=str, help="PDK token for accessing data")
    #parser.add_argument('start_date', type=str, help="the start date for parsing data (yyyy-mm-dd format)")
    #parser.add_argument('end_date', type=str, help="the end date for parsing data (yyyy-mm-dd format)")
    
    args = parser.parse_args()

    ids = []
    with open(args.id_file, "rb") as wave_f:
        for line in wave_f.readlines():
            ids.append(line.strip())

    generators = []
    with open(args.sensor_file, "rb") as sensor_f:
        for line in sensor_f.readlines():
            generators.append(line.strip())

    process_gen_partial = partial(process_generators, 
                                  generators=generators, 
                                  data_source=args.data_source,
                                  out_loc=args.out_loc,
                                  start_date=args.start_date,
                                  end_date=args.end_date,
                                  debug=args.debug
                                  )

    # bug in Python 2.7.x multiprocessing requires hack for graceful keyboard interrupt
    # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = Pool(processes=args.num_procs)
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        res = pool.map_async(process_gen_partial, ids)
        # one year in seconds
        res.get(31556952)
    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close() 
    pool.join()