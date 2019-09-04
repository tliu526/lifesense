"""
Offline script for PDK data pull
"""
import argparse
import json
import pickle

from datetime import date, timedelta
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from pdk_client import PDKClient

# client setup
SITE_URL = "https://lifesense.fsm.northwestern.edu/data"
TOKEN = "oCZUlrEtmlJRsy9b8fOpNTthlMPT7kU6GcniFLt0yLH4Yz0ExkGeflPuaPlOwCyj"
PAGE_SIZE = 100
client = PDKClient(site_url=SITE_URL, token=TOKEN)

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
    """Processes counts for the given id over the date range.

    Dumps the resulting DataFrame into the specified directory.

    TODO actually implement the above parameters

    """
    id_df = pd.DataFrame()
    print(id)
    start_date = date(2019, 8, 23)
    end_date = date(2019, 8, 24)
    cur_date = start_date
    while cur_date <= end_date:
        start_str = cur_date.strftime("%Y-%m-%d")
        end_str = (cur_date + timedelta(days=1)).strftime("%Y-%m-%d")
        d = get_data_counts(id, generators, start_str, end_str)
        df = pd.DataFrame(d, index=[0])
        df['date'] = cur_date
        df['pid'] = id
        id_df = id_df.append(df)
        cur_date += timedelta(days=1) 

    pickle.dump(id_df, open("data_pull/generator_counts2/{}.df".format(id), 'wb'), -1)

if __name__ == '__main__':
    wave1_ids = []
    with open("data_pull/wave1_ids.txt", "rb") as wave_f:
        for line in wave_f.readlines():
            wave1_ids.append(line.strip())

    pool = Pool(processes=6)
    pool.map(process_counts, wave1_ids)
    pool.close() 