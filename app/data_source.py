import ray
import os

ray.init(runtime_env={'working_dir': ".", 'pip': "requirements.txt",
                      'env_vars': dict(os.environ),
                      'excludes': ['*.jar', '.git*/', 'jupyter/']}) if not ray.is_initialized() else True
import pandas as pd
import app.feature_store as feature_store
import pytz
from datetime import datetime
import logging
# from app.main.python.utils import utils

data_columns = None


def get_data(begin_offset=None, end_offset=None, new_data=None):
    data = get_cached_data()

    if data is None:
        csv_data_source_path = 'data/airlinetweets.csv'
        data = pd.read_csv(csv_data_source_path, parse_dates=['tweet_created'],
                           index_col=['tweet_created']).sort_index()

        # Adjust timestamps to align with today's date for demo purposes
        current_dt = pytz.utc.localize(datetime.now())
        lag_adjustment = current_dt - data.index.max()
        data.set_index(data.index + lag_adjustment, inplace=True)

        # Temporary: Get only the last 7 days
        # (based on the state of the data in the csv file)
        # data = data[data.index > (current_dt - timedelta(days=7))]

        # Store the last published date as the offset
        last_published_date = current_dt
        store_global_offset(last_published_date)
    else:
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S%z', utc=True, errors='coerce')

    if new_data is not None:
        data = pd.concat([data, new_data])

    if begin_offset is not None:
        # Fetch new rows starting from after the begin offset
        data = data[data.index > begin_offset]

    if end_offset is not None:
        # New rows should come before the end offset
        data = data[data.index < end_offset]

    # Store to feature store
    feature_store.save_artifact(data, '_data', distributed=False)

    return data


def get_cached_data():
    return feature_store.load_artifact('_data', distributed=False)


def add_data(new_data):
    return get_data(new_data=new_data)


def get_data_schema():
    global data_columns
    if data_columns is None:
        csv_data_source_path = 'data/airlinetweets.csv'
        data_columns = pd.read_csv(csv_data_source_path, parse_dates=['tweet_created'],
                                   index_col=['tweet_created']).columns
    return data_columns


def store_global_offset(dt):
    offset = int(dt.timestamp())
    logging.info(f"saving original offset...{offset}")

    # save offset to global store
    feature_store.save_offset(offset, 'original')
    feature_store.save_offset(dt, 'original_datetime')

    # update all relevant consumers to read from the original offset
    # monitors = [config.firehose_monitor]
    # for monitor in monitors:
    #     monitor.read_data(offset)
