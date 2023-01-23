########################
# Imports
########################
import pandas as pd
import numpy as np
import logging

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import sklearn.model_selection as model_selection
from scdfutils import utils
from prodict import Prodict
from mlmetrics import exporter
from datetime import datetime, timedelta, timezone
import pytz
import json
import ray
import os
from distributed.ray.distributed import ScaledTaskController

ray.init(runtime_env={'working_dir': ".", 'pip': "requirements.txt",
                      'env_vars': dict(os.environ),
                      'excludes': ['*.jar', '.git*/', 'jupyter/']}) if not ray.is_initialized() else True


########################
# Ingest Data
########################
def ingest_data():
    logging.info('Ingest data...')
    data = pd.read_csv('data/sentiment.csv', parse_dates=['tweet_created'], index_col=['tweet_created']).sort_index()
    data.index = utils.index_as_datetime(data)
    return data


########################################################################################################################
# SENTIMENT ANALYSIS
########################################################################################################################


#############################
# Prepare Data
#############################
def prepare_data(df):
    logging.info("Preparing data...")
    df = feature_extraction(df)
    df = feature_encoding(df)
    return df


#############################
# Perform Feature Encoding
#############################
def feature_encoding(df):
    logging.info("Performing feature encoding...")
    target_map = {'positive': 1, 'negative': 0, 'neutral': 2}
    df['target'] = df['airline_sentiment'].map(target_map)
    return df


#############################
# Perform Feature Extraction
#############################
def feature_extraction(df):
    logging.info("Performing feature extraction...")
    return df[['airline_sentiment', 'text']].copy()


#############################
# Apply Train-Test Split
#############################
def train_test_split(df):
    logging.info("Performing train/test data split...")
    df_train, df_test = model_selection.train_test_split(df)
    return df_train, df_test


#############################
# Apply Data Vectorization
#############################
def vectorization(df_train, df_test):
    logging.info("Preparing data vectorization (tf-idf encoding)...")
    vectorizer = TfidfVectorizer(max_features=2000)
    x_train = vectorizer.fit_transform(df_train['text'])
    x_test = vectorizer.transform(df_test['text'])
    y_train = df_train['target']
    y_test = df_test['target']
    return x_train, x_test, y_train, y_test, vectorizer


########################
# Train
########################
def train(x_train, x_test, y_train, y_test):
    logging.info("Training data...")
    utils.mlflow_generate_autolog_metrics('sklearn')
    model = LogisticRegression(max_iter=500)  # TODO: try different values of C, penalty
    model.fit(x_train, y_train)
    generate_and_save_metrics(x_train, x_test, y_train, y_test, model)
    return model


########################
# Generate Metrics
########################
def generate_and_save_metrics(x_train, x_test, y_train, y_test, model):
    logging.info("Generating metrics...")
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    train_roc_auc = roc_auc_score(y_train, model.predict_proba(x_train), multi_class='ovo')
    test_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test), multi_class='ovo')

    logging.info("Saving metrics...")
    parent_run_id = utils.get_root_run_id(experiment_names=[utils.get_env_var('CURRENT_EXPERIMENT')])
    utils.mlflow_log_metric(parent_run_id, key='sentiment_train_acc', value=train_acc)
    utils.mlflow_log_metric(parent_run_id, key='sentiment_test_acc', value=test_acc)
    utils.mlflow_log_metric(parent_run_id, key='sentiment_train_roc_auc', value=train_roc_auc)
    utils.mlflow_log_metric(parent_run_id, key='sentiment_test_roc_auc', value=test_roc_auc)

    # Publish ML metrics
    logging.info(f"Exporting ML metrics - sentiment_train_acc...{train_acc}, sentiment_test_acc...{test_acc}, "
                 f"sentiment_train_roc_auc...{train_roc_auc}, sentiment_test_roc_auc...{test_roc_auc}")
    scdf_tags = Prodict.from_dict(json.loads(utils.get_env_var('SCDF_RUN_TAGS')))
    exporter.prepare_histogram('sentimentanalysis:trainacc', 'Sentiment Train ACC', scdf_tags, train_acc)
    exporter.prepare_histogram('sentimentanalysis:testacc', 'Sentiment Test ACC', scdf_tags, test_acc)
    exporter.prepare_histogram('sentimentanalysis:trainaucroc', 'Sentiment Train AUC-ROC', scdf_tags, train_roc_auc)
    exporter.prepare_histogram('sentimentanalysis:testacc', 'Sentiment Test AUC-ROC', scdf_tags, test_roc_auc)


########################
# Save Model
########################
def save_model(model, artifact_path=None):
    logging.info("Saving model...")
    controller = ScaledTaskController.remote()
    controller.log_model.remote(
        utils.get_root_run_id(experiment_names=[utils.get_env_var('CURRENT_EXPERIMENT')]),
        model,
        'sklearn',
        artifact_path=artifact_path or 'sklearn',
        registered_model_name='sentiment_analysis_model',
        await_registration_for=None)


########################
# Save Vectorizer
########################
def save_vectorizer(vectorizer):
    controller = ScaledTaskController.remote()
    logging.info("Saving vectorizer...")
    parent_run_id = utils.get_root_run_id(experiment_names=[utils.get_env_var('CURRENT_EXPERIMENT')])
    controller.log_artifact.remote(parent_run_id, vectorizer, '/parent/app/artifacts')


########################
# Predict Sentiment
########################
def predict(text, stage='None'):
    logging.info("Predicting sentiment...")
    controller = ScaledTaskController.remote()
    sample = pd.Series(text)
    parent_run_id = utils.get_root_run_id(experiment_names=[utils.get_env_var('CURRENT_EXPERIMENT')])
    vectorizer = controller.load_artifact.remote(parent_run_id, 'sentiment_analysis_vectorizer')
    model = controller.load_model.remote(parent_run_id, 'sklearn', model_uri=f'models:/baseline_model/{stage}')
    transformed_sample = vectorizer.transform(sample)
    classes = ['negative', 'positive', 'neutral']
    return classes[model.predict(transformed_sample)[0]]


def generate_dummy_model_data(num_classes=2, size=1000):
    x = np.linspace(0, np.pi * 8, num=size)
    y = np.random.randint(0, num_classes, int(size / num_classes))
    dataset = pd.DataFrame(data={'x': x, 'target': y})
    now = pytz.utc.localize(datetime.now())
    dataset.index = dataset.index.map(lambda i: now + timedelta(minutes=i))
    return dataset
