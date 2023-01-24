from scdfutils import utils, ports
import logging
from scdfutils.http_status_server import HttpHealthServer
from mlmetrics import exporter
import mlflow
from sklearn.dummy import DummyClassifier
from distributed.ray.distributed import ScaledTaskController
from prodict import Prodict
import json
from datetime import datetime
import app.sentiment_analysis
import os
from scdfutils.run_adapter import scdf_adapter
import ray
import app.feature_store as feature_store
import app.anomaly_detection_arima as anomaly_detection_arima
import app.anomaly_detection_rnn as anomaly_detection_rnn
import traceback

HttpHealthServer.run_thread()
logger = logging.getLogger('mlmodeltest')
ray.init(runtime_env={'working_dir': ".", 'pip': "requirements.txt",
                      'env_vars': dict(os.environ),
                      'excludes': ['*.jar', '.git*/', 'jupyter/']}) if not ray.is_initialized() else True
buffer = None
dataset = None


@scdf_adapter(environment=None)
def process_sentiment_model(msg):
    logger.info("in process step...")
    global dataset, buffer
    controller = ScaledTaskController.remote()
    if buffer is None:
        logger.info("Preloading data...")
        buffer = ray.data.from_items(utils.get_csv_rows('./data/preload.csv'))
        logger.info("Data preloaded.")
    buffer = buffer.union(ray.data.from_items([msg.split(',')]))

    ready = buffer.count() > (utils.get_env_var('MONITOR_SLIDING_WINDOW_SIZE') or 200)
    run_id = utils.get_env_var('MLFLOW_RUN_ID')
    experiment_id = utils.get_env_var('MLFLOW_EXPERIMENT_ID')
    parent_run_id = utils.get_root_run_id(experiment_names=[utils.get_env_var('CURRENT_EXPERIMENT')])

    # Print MLproject parameter(s)
    logger.info(
        f"MLflow parameters: ready={ready}, msg={msg}, run_id={run_id}, experiment_id={experiment_id}, parent_run_id={parent_run_id}")

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=datetime.now().strftime("%Y-%m-%d-%H%M"),
                          nested=True):
        #######################################################
        # BEGIN processing
        #######################################################

        # Once the window size is large enough, start processing
        if ready:
            msgs = buffer.take_all()
            dataset = utils.initialize_timeseries_dataframe(msgs, 'data/schema.csv')
            dataset = app.sentiment_analysis.prepare_data(dataset)

            # Perform Test-Train Split
            df_train, df_test = app.sentiment_analysis.train_test_split(dataset)

            # Perform tf-idf vectorization
            x_train, x_test, y_train, y_test, vectorizer = app.sentiment_analysis.vectorization(df_train, df_test)

            # Generate model
            baseline_model = app.sentiment_analysis.train(x_train, x_test, y_train, y_test)

            # Store metrics
            app.sentiment_analysis.generate_and_save_metrics(x_train, x_test, y_train, y_test, baseline_model)

            # Save model
            app.sentiment_analysis.save_model(baseline_model)

            # Save vectorizer
            app.sentiment_analysis.save_vectorizer(vectorizer)

            # Upload artifacts
            controller.log_artifact.remote(parent_run_id, dataset, 'dataset_snapshot')

            #######################################################
            # RESET globals
            #######################################################
            buffer = None
            dataset = None
        else:
            logger.info(
                f"Buffer size not yet large enough to process: expected size {utils.get_env_var('MONITOR_SLIDING_WINDOW_SIZE') or 200}, actual size {buffer.count()} ")
        logger.info("Completed process step.")

        #######################################################
        # END processing
        #######################################################

        return ready


@scdf_adapter(environment=None)
def evaluate_sentiment_model(ready):
    controller = ScaledTaskController.remote()
    run_id = utils.get_env_var('MLFLOW_RUN_ID')
    experiment_id = utils.get_env_var('MLFLOW_EXPERIMENT_ID')
    parent_run_id = utils.get_root_run_id(experiment_names=[utils.get_env_var('CURRENT_EXPERIMENT')])
    model_name = 'sentiment_analysis_model'
    flavor = 'sklearn'

    # Print MLproject parameter(s)
    logger.info(
        f"MLflow parameters: ready={ready}, run_id={run_id}, experiment_id={experiment_id}, parent_run_id={parent_run_id}")

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=datetime.now().strftime("%Y-%m-%d-%H%M"),
                          nested=True):
        #######################################################
        # BEGIN processing
        #######################################################
        # Once the data is ready, start processing
        if ready:

            # Load existing baseline model (or generate dummy regressor if no model exists)
            version = utils.get_latest_model_version(name=model_name, stages=['None', 'Staging'])

            if version:
                # Get candidate model
                candidate_model = ray.get(controller.load_model.remote(
                    parent_run_id,
                    flavor,
                    model_uri=f'models:/{model_name}/{version}'))

                # Get baseline model
                if version > 0:
                    baseline_model = ray.get(controller.load_model.remote(
                        parent_run_id,
                        flavor,
                        model_uri=f'models:/{model_name}/{version - 1}'))
                else:
                    dummy_data = app.sentiment_analysis.generate_dummy_model_data(num_classes=3, size=1000)
                    baseline_model = DummyClassifier(strategy="uniform").fit(dummy_data['x'], dummy_data['target'])

                # Get validation data
                data = ray.get(controller.load_artifact.remote(parent_run_id,
                                                               'dataset_snapshot',
                                                               artifact_uri=f"runs:/{parent_run_id}/dataset_snapshot",
                                                               dst_path="/parent/app/artifacts"))
                data.index = utils.index_as_datetime(data)

                # if model evaluation passes, promote candidate model to "staging", else retain baseline model
                logging.info(
                    f"Evaluating baseline vs candidate models: baseline_model={baseline_model}, candidate_model={candidate_model}, version={version}")
                result = controller.evaluate_models.remote(baseline_model=baseline_model,
                                                           candidate_model=candidate_model,
                                                           data=data, version=version)
                logger.info("Get remote results...")
                evaluation_result = ray.get(result)
                logger.info(f"Evaluation result: {evaluation_result}")

                # Publish ML metrics
                scdf_tags = Prodict.from_dict(json.loads(utils.get_env_var('SCDF_RUN_TAGS')))
                scdf_tags = Prodict.from_dict({**scdf_tags, **{'model_name': 'sentimentanalysis'}})
                exporter.prepare_counter('deploynotification',
                                         'New Candidate Model Readiness Notification', scdf_tags,
                                         int(evaluation_result))

            else:
                logger.error("Baseline model not found...could not perform evaluation")

        else:
            logger.info(f"Data not yet available for processing.")

        logger.info("Completed evaluate step.")
        #######################################################
        # END processing
        #######################################################

        return dataset


@scdf_adapter(environment=None)
def process_anomaly_arima_model(sample_frequency, reporting_timeframe, rebuild='False'):
    return process_anomaly_model().remote(sample_frequency, reporting_timeframe, rebuild=rebuild, model_type=anomaly_detection_arima)


@scdf_adapter(environment=None)
def process_anomaly_rnn_model(sample_frequency, reporting_timeframe, rebuild='False'):
    return process_anomaly_model().remote(sample_frequency, reporting_timeframe, rebuild=rebuild, model_type=anomaly_detection_rnn)


@ray.remote(num_cpus=2, memory=40 * 1024 * 1024)
def process_anomaly_model(sample_frequency, reporting_timeframe, rebuild='False', model_type=anomaly_detection_arima):
    rebuild = eval(rebuild)

    # Input features
    data_freq, sliding_window_size, estimated_seasonality_hours, arima_order, training_percent = 10, 144, 24, None, 0.73  # 0.80
    logging.info(
        f"Params: data_freq={data_freq}, sliding_window_size={sliding_window_size}, training_percent={training_percent}")

    # Other required variables

    extvars = model_type.get_utility_vars()

    # Set up metrics

    try:
        # Ingest Data
        df = model_type.ingest_data()

        # Store input values
        model_type.initialize_input_features(data_freq, sliding_window_size, arima_order)

        # Prepare data by performing feature extraction
        buffers = model_type.prepare_data(df, sample_frequency, extvars)

        # Determine the training window
        num_future_predictions = 2
        if rebuild:
            total_training_window = int(training_percent * len(buffers['actual_negative_sentiments']))
            total_forecast_window = len(
                buffers['actual_negative_sentiments']) - total_training_window + num_future_predictions
        else:
            total_training_window = len(buffers['actual_negative_sentiments'])
            total_forecast_window = num_future_predictions

        # Save EDA artifacts
        model_type.generate_and_save_eda_metrics(df)

        # Perform ADF test
        adf_results = model_type.generate_and_save_adf_results(
            buffers['actual_negative_sentiments'])
        model_type.generate_and_save_stationarity_results(buffers['actual_negative_sentiments'],
                                                          estimated_seasonality_hours)

        # Check for stationarity
        logging.info(f'Stationarity : {model_type.check_stationarity(adf_results)}')
        logging.info(f'P-value : {adf_results[1]}')

        # Build a predictive model (or reuse existing one if this is not rebuild mode)
        stepwise_fit = model_type.build_model(buffers['actual_negative_sentiments'], rebuild)

        # Perform training
        model_results = model_type.train_model(total_training_window, stepwise_fit,
                                               buffers['actual_negative_sentiments'],
                                               rebuild=rebuild,
                                               data_freq=data_freq)

        # Perform forecasting
        model_forecasts = model_type.generate_forecasts(sliding_window_size,
                                                        total_forecast_window,
                                                        stepwise_fit,
                                                        buffers[
                                                            'actual_negative_sentiments'],
                                                        rebuild,
                                                        total_training_window=total_training_window)

        # Detect anomalies
        model_results_full = model_type.detect_anomalies(model_results,  # fittedvalues,
                                                         total_training_window,
                                                         buffers['actual_negative_sentiments'])

        # Plot anomalies
        fig = model_type.plot_trend_with_anomalies(buffers['actual_negative_sentiments'],
                                                   model_results_full,
                                                   model_forecasts,
                                                   total_training_window,
                                                   stepwise_fit,
                                                   extvars,
                                                   reporting_timeframe,
                                                   data_freq=data_freq)

        # TEMPORARY: Set a flag indicating that training was done
        feature_store.save_artifact(True, 'anomaly_detection_is_trained', distributed=False)

        return True

    except Exception as e:
        logging.error('Could not complete execution - error occurred: ', exc_info=True)
        traceback.print_exc()


@scdf_adapter(environment=None)
def evaluate_anomaly_arima_model(ready):
    run_id = utils.get_env_var('MLFLOW_RUN_ID')
    experiment_id = utils.get_env_var('MLFLOW_EXPERIMENT_ID')
    parent_run_id = utils.get_root_run_id(experiment_names=[utils.get_env_var('CURRENT_EXPERIMENT')])
    model_name = 'anomaly_arima_model'

    # Print MLproject parameter(s)
    logger.info(
        f"MLflow parameters: ready={ready}, run_id={run_id}, experiment_id={experiment_id}, parent_run_id={parent_run_id}")

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=datetime.now().strftime("%Y-%m-%d-%H%M"),
                          nested=True):
        if ready:
            return True  # no-op
    return False


@scdf_adapter(environment=None)
def evaluate_anomaly_rnn_model(ready):
    run_id = utils.get_env_var('MLFLOW_RUN_ID')
    experiment_id = utils.get_env_var('MLFLOW_EXPERIMENT_ID')
    parent_run_id = utils.get_root_run_id(experiment_names=[utils.get_env_var('CURRENT_EXPERIMENT')])
    model_name = 'anomaly_rnn_model'

    # Print MLproject parameter(s)
    logger.info(
        f"MLflow parameters: ready={ready}, run_id={run_id}, experiment_id={experiment_id}, parent_run_id={parent_run_id}")

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=datetime.now().strftime("%Y-%m-%d-%H%M"),
                          nested=True):
        if ready:
            return True  # no-op
    return False
