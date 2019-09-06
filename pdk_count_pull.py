"""
Python script for PDK data pull.

Requires python 2.7, for the pdk_client module

"""
import argparse
import json

from datetime import date, timedelta, datetime
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from pdk_client import PDKClient

DATE_FMT = "%Y-%m-%d"

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
        gen_query = query.filter(generator_identifier=generator,
                            created__gte=start_date,
                            created__lt=end_date)
        count_dict[generator] = gen_query.count()

    return count_dict

def process_counts(id, start_date, end_date, out_dir):
    """Processes counts for the given id over the date range

    Dumps the resulting DataFrame into the specified directory.

    Args:
        id (str): the participant id to pull
        start_date (str): the start date in yyyy-mm-dd form
        end_date (str): the start date in yyyy-mm-dd form
        out_dir (str): the name of the output directory

    Returns:
        None
    """
    id_df = pd.DataFrame()
    print(id)
    start_date = datetime.strptime(start_date, DATE_FMT)
    end_date = datetime.strptime(end_date, DATE_FMT)
    cur_date = start_date
    while cur_date <= end_date:
        #print(cur_date)
        start_str = cur_date.strftime(DATE_FMT)
        end_str = (cur_date + timedelta(days=1)).strftime(DATE_FMT)
        d = get_data_counts(id, generators, start_str, end_str)
        df = pd.DataFrame(d, index=[0])
        df['date'] = cur_date
        df['pid'] = str(id)
        id_df = id_df.append(df)
        cur_date += timedelta(days=1) 

    id_df.to_csv("{}/{}.csv".format(out_dir, id), index=False)

if __name__ == '__main__':
    script_description = "Process PDK sensor counts and dumps them to .csv for processing.\nExample Usage: python pdk_count_pull.py wave1_ids.txt pdk_counts/ <pdk_token_string> 2019-08-31 2019-09-05 8"
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('id_file', type=str, help="a file containing participant ids, separated by newlines")
    parser.add_argument('out_dir', type=str, help="the output directory location")
    parser.add_argument('pdk_token', type=str, help="PDK token for accessing data")
    parser.add_argument('start_date', type=str, help="the start date for parsing data (yyyy-mm-dd format)")
    parser.add_argument('end_date', type=str, help="the end date for parsing data (yyyy-mm-dd format)")
    parser.add_argument('num_procs', type=int, default=2, help="the number of processes to allocate (default 2)")
    
    args = parser.parse_args()

    ids = []
    with open(args.id_file, "rb") as wave_f:
        for line in wave_f.readlines():
            ids.append(line.strip())

    # client setup, TODO parameterize
    SITE_URL = "https://lifesense.fsm.northwestern.edu/data"
    TOKEN = args.pdk_token
    PAGE_SIZE = 100
    client = PDKClient(site_url=SITE_URL, token=TOKEN)

    # have to use partials since python 2.7 multiprocessing doesn't have starmap()
    process_counts_partial = partial(process_counts,
                                     start_date = args.start_date,                                     
                                     end_date = args.end_date,
                                     out_dir  = args.out_dir)                                     

    pool = Pool(processes=args.num_procs)
    pool.map(process_counts_partial, ids)
    pool.close() 

    count_df = pd.DataFrame()
    for pid in ids:
        df = pd.read_csv("{}/{}.csv".format(args.out_dir, pid))
        count_df = count_df.append(df)
    count_df.to_csv("{}/generator_counts_{}-{}.csv".format(args.out_dir, args.start_date, args.end_date), index=False)