########################
# Imports
########################
import ray
import os

ray.init(runtime_env={'working_dir': ".", 'pip': "requirements.txt",
                      'env_vars': dict(os.environ),
                      'excludes': ['*.jar', '.git*/', 'jupyter/']}) if not ray.is_initialized() else True
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import scipy.stats as st
import re
import math
import json
import feature_store, data_source, anomaly_detection
import distributed.ray.utilities as utils_ext
from mlmetrics import exporter
import Prodict


########################################################################################################################
# ANOMALY DETECTION
########################################################################################################################


########################
# Ingest Data
########################
def ingest_data():
    return anomaly_detection.ingest_data()


#######################################
# Set Global Values
#######################################
def initialize_input_features(data_freq, sliding_window_size, arima_order):
    return anomaly_detection.initialize_input_features(data_freq, sliding_window_size, arima_order)


#######################################
# Generate and Save EDA Artifacts
#######################################
def generate_and_save_eda_metrics(df):
    return anomaly_detection.generate_and_save_eda_metrics(df)


#############################
# Filter Data
#############################
def filter_data(df, head=True, num_rows_head=None, num_rows_tail=None):
    return anomaly_detection.filter_data(df, head, num_rows_head, num_rows_tail)


#############################
# Prepare Data
#############################

def prepare_data(df, sample_frequency, extvars):
    return anomaly_detection.prepare_data(df, sample_frequency, extvars)


#######################################
# Perform data cleansing
#######################################
def cleanse_data(df):
    return anomaly_detection.cleanse_data(df)


#######################################
# Perform feature extraction via resampling
#######################################
def extract_features(df, sample_frequency='10min', extvars={}, is_partial_data=False):
    return anomaly_detection.extract_features(df, sample_frequency, extvars, is_partial_data)


#######################################
# Perform Data Standardization
#######################################
def standardize_data(buffers, extvars):
    return anomaly_detection.standardize_data(buffers, extvars)


#######################################
# Initialize Data Buffers
#######################################
def get_filtered_data_sets(df, sample_frequency, extvars):
    return anomaly_detection.get_filtered_data_sets(df, sample_frequency, extvars)


#######################################
# Generate and Save ADF Results
#######################################
def generate_and_save_adf_results(actual_negative_sentiments):
    logging.info("Generate and save Dickey-Fuller test results...")
    adfuller_results = adfuller(actual_negative_sentiments['sentiment_normalized'])
    feature_store.save_artifact(adfuller_results, 'adf_results')
    return adfuller_results


#######################################
# Check for stationarity
#######################################
def check_stationarity(adfuller_results):
    logging.info("Check for stationarity...")
    return adfuller_results[1] < 0.05


#######################################
# Generate and Save Stationary Results
#######################################
def generate_and_save_stationarity_results(actual_negative_sentiments, sliding_window_size):
    logging.info("Save stationarity plot results...")
    plot_acf(actual_negative_sentiments['sentiment'], lags=20)
    plt.savefig("anomaly_acf.png", bbox_inches='tight')
    plot_pacf(actual_negative_sentiments['sentiment_normalized'])
    plt.savefig("anomaly_pacf.png", bbox_inches='tight')
    seasonal_decompose(actual_negative_sentiments['sentiment'], model='additive', period=sliding_window_size).plot()
    plt.savefig("anomaly_seasonal_decompose.png", bbox_inches='tight')


#######################################
# Perform Auto ARIMA to build model
#######################################
def build_model(actual_negative_sentiments, rebuild=False):
    logging.info("Running auto_arima to build ARIMA model...")
    stepwise_fit = feature_store.load_artifact('anomaly_auto_arima', distributed=False)

    if rebuild is True:
        stepwise_fit = auto_arima(actual_negative_sentiments['sentiment_normalized'], start_p=0, start_q=0, max_p=6,
                                  max_q=6,
                                  seasonal=True, trace=True)

    logging.info(f"stepwise fit is now...{stepwise_fit}")

    feature_store.save_artifact(stepwise_fit, 'anomaly_auto_arima', distributed=False)
    return stepwise_fit


#######################################
# Train ARIMA Model To Generate Results
#######################################
def train_model(training_window_size, stepwise_fit, actual_negative_sentiments, rebuild=False,
                data_freq=10):
    logging.info(f"Train ARIMA model with params (p,d,q) = {stepwise_fit.order}...")
    actual_negative_sentiments_train = actual_negative_sentiments.iloc[:int(training_window_size)]

    model_arima_order = stepwise_fit.order

    model_arima = ARIMA(actual_negative_sentiments_train['sentiment_normalized'], order=model_arima_order)

    feature_store.save_artifact(model_arima, 'anomaly_arima_model', distributed=False)

    model_arima_results = model_arima.fit()  # fit the model

    feature_store.log_metric(model_arima_results.aic, 'aic', distributed=False)  # track the Akaike Information Criterion

    feature_store.log_metric(model_arima_results.mae, 'mae', distributed=False)  # track the Mean Absolute Error

    # Publish ML metrics
    logging.info(f"Exporting ML metrics - AIC...{model_arima_results.aic}, BIC...{model_arima_results.mae}, ")
    scdf_tags = Prodict.from_dict(json.loads(utils_ext.get_env_var('SCDF_RUN_TAGS')))
    scdf_tags = Prodict.from_dict({**scdf_tags, **{'model_type': 'arima'}})
    exporter.prepare_histogram('anomalydetection:aic', 'Akaike Information Criterion', scdf_tags, model_arima_results.aic)
    exporter.prepare_histogram('anomalydetection:mae', 'Mean Absolute Error', scdf_tags, model_arima_results.mae)

    return model_arima_results.fittedvalues


#######################################
# Test ARIMA Model
#######################################
def test_arima_model(sliding_window_size, total_forecast_size, stepwise_fit, actual_negative_sentiments):
    logging.info('Testing ARIMA model...')

    return generate_forecasts(sliding_window_size, total_forecast_size, stepwise_fit,
                              actual_negative_sentiments)


#######################################
# Detect Anomalies
#######################################
def detect_anomalies(predictions, window_size, actual_negative_sentiments):
    logging.info('Detecting anomalies...')

    z_score = st.norm.ppf(.95)  # 95% confidence interval
    mae_scale_factor = 0.67449  # MAE is 0.67449 * std

    predictions = predictions.iloc[-int(window_size):]

    df_total = actual_negative_sentiments['sentiment_normalized'].iloc[:int(window_size)].iloc[-len(predictions):]
    # mae = median_absolute_error(df_total.iloc[-int(window_size):], predictions)
    logging.info(f"anomalies for...{df_total} {predictions}")
    mae = median_absolute_error(df_total, predictions)

    model_arima_results_full = \
        pd.DataFrame({'fittedvalues': predictions, 'median_values': predictions.rolling(4).median().fillna(0)},
                     index=predictions.index)
    model_arima_results_full['threshold'] = model_arima_results_full['median_values'] + (
            z_score / mae_scale_factor) * mae
    model_arima_results_full['anomaly'] = 0

    model_arima_results_full['actualvalues'] = df_total
    model_arima_results_full['actualvalues'].fillna(0, inplace=True)

    model_arima_results_full.loc[
        model_arima_results_full['actualvalues'] > model_arima_results_full['threshold'], 'anomaly'] = 1

    print(f"Anomaly distribution: \n{model_arima_results_full['anomaly'].value_counts()}")

    # TODO: Publish anomaly summary to queue
    feature_store.save_artifact(actual_negative_sentiments, 'actual_negative_sentiments', distributed=False)
    publish_trend_stats(actual_negative_sentiments)

    return model_arima_results_full


#######################################
# Generate ARIMA Forecasts
#######################################
def generate_forecasts(sliding_window_size, total_forecast_size, stepwise_fit, actual_negative_sentiments,
                       rebuild=False,
                       total_training_window=144):
    logging.info("Generate ARIMA predictions...")

    # The number of forecasts per sliding window will be the number of AR or MA lags, as ARIMA can't forecast beyond that
    num_lags = max(stepwise_fit.order[0], max(stepwise_fit.order[2], 1))

    # The number of sliding windows will be ( total forecast size / num_lags )
    num_sliding_windows = math.ceil(total_forecast_size / num_lags)

    # The dataset to forecast with
    if rebuild:
        num_shifts = total_training_window + total_forecast_size - len(actual_negative_sentiments)
        df = actual_negative_sentiments.iloc[:int(total_training_window)]
        df = utils_ext.get_next_rolling_window(df, num_shifts) if num_shifts else df
    else:
        df = utils_ext.get_next_rolling_window(actual_negative_sentiments, total_forecast_size)

    # Initialize the start & end indexes
    end_idx = len(df) - num_lags

    # Get any prior ARIMA forecasts
    predictions = get_prior_forecasts()

    for idx in np.arange(num_sliding_windows):
        # Compute the start & end indexes
        end_idx = end_idx + num_lags
        start_idx = end_idx - sliding_window_size
        logging.info(f'DEBUG: {start_idx} {end_idx} {end_idx - start_idx} {len(df)} {len(actual_negative_sentiments)}')
        tmp_data = actual_negative_sentiments[int(start_idx):int(end_idx)]
        tmp_arima = ARIMA(tmp_data['sentiment_normalized'], order=stepwise_fit.order)
        tmp_model_arima_results = tmp_arima.fit()
        pred = tmp_model_arima_results.forecast(steps=num_lags, typ="levels").rename('forecasted')
        predictions = predictions.append(pd.Series(pred))

    # Save forecasts
    logging.info(f"Number of anomaly_arima_forecasts to save...{len(predictions)}")
    feature_store.save_artifact(predictions, 'anomaly_arima_forecasts')

    # Return predictions
    return predictions


#######################################
# Get any prior forecasts
#######################################

def get_prior_forecasts():
    forecasts = feature_store.load_artifact('anomaly_arima_forecasts')
    logging.info(f"Number of forecasts loaded...{len(forecasts) if forecasts is not None else 0}")
    if forecasts is None:
        forecasts = pd.Series([])
    return forecasts


##############################################
# Get latest predictions from prior forecasts
##############################################

def get_predictions_before_or_at(dt):
    forecasts = feature_store.load_artifact('anomaly_arima_forecasts')
    logging.info(f"forecasts is {dt} {forecasts}")
    if forecasts is None:
        return pd.Series([])
    return forecasts[forecasts.index <= dt]


##############################################
# Get latest forecasts
##############################################

def get_forecasts_after(dt):
    forecasts = feature_store.load_artifact('anomaly_arima_forecasts')
    logging.info(f"Number of forecasts loaded...{len(forecasts) if forecasts is not None else 0}")
    if forecasts is None:
        return pd.Series([])
    return forecasts[forecasts.index > dt]


#######################################
# Convert timeframe flag to number of time lags
#######################################
def get_time_lags(timeframe='day'):
    logging.info(f"Get time lag for {timeframe}...")
    time_lags = {'hour': 1, 'day': 24, 'week': 168}
    return time_lags[timeframe]


#######################################
# Generate and publish stats
#######################################


def publish_trend_stats(actual_negative_sentiments=None):
    if actual_negative_sentiments is None:
        actual_negative_sentiments = feature_store.load_artifact('actual_negative_sentiments', distributed=False, can_cache=False)

    sample_frequencies = ['1min', '10min', '60min']

    stats = []

    old_summary = feature_store.load_artifact('anomaly_summary', distributed=False)
    if old_summary is None:
        old_summary = pd.DataFrame()

    for sample_frequency in sample_frequencies:
        num_negative_in_past, sample_frequency_num = 0, int(re.findall(r'\d+', sample_frequency)[0])

        last_recorded_time = actual_negative_sentiments.index[-1]
        offset_time = last_recorded_time - timedelta(minutes=sample_frequency_num)

        num_negative_in_past = actual_negative_sentiments.loc[actual_negative_sentiments.index >= offset_time][
            'sentiment'].sum()
        logging.info(f"Number of negative posts in past {sample_frequency_num} minutes: {num_negative_in_past}")

        stats.append(num_negative_in_past)

    new_summary = {sample_frequencies[i]: [stats[i]] for i in range(len(sample_frequencies))}
    new_summary['anomaly_found'] = [False]  # TODO: Set to True if new anomaly is found
    new_summary = pd.DataFrame.from_dict(new_summary)

    summary = pd.concat([old_summary, new_summary])
    logging.info(f"New Summary: {new_summary}")

    feature_store.save_artifact(summary, 'anomaly_summary', distributed=False)

    return new_summary


#######################################
# Retrieve generated stats
#######################################


def get_trend_stats():
    return feature_store.load_artifact('anomaly_summary', distributed=False)


def process_stats(head, body):
    logging.info('In process_stats...')
    logging.info(f'{json.loads(body)} {head}')


#######################################
# Get utility variables
# (data normalizers, etc)
#######################################
def get_utility_vars():
    return {
        'anomaly_positive_standard_scalar': StandardScaler(),
        'anomaly_neutral_standard_scalar': StandardScaler(),
        'anomaly_negative_standard_scalar': StandardScaler(),
    }


#######################################
# Utility: Check if retraining is required
# (data normalizers, etc)
#######################################
# TODO: Use external pipeline like Argo Workflow/Airflow/Spring Cloud Data Flow
def anomaly_detection_needs_training():
    return feature_store.load_artifact('anomaly_detection_arima_is_trained', distributed=False) is None
