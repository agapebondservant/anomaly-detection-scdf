import joblib
import logging
from distributed.ray.distributed import ScaledTaskController
import distributed.ray.utilities as utils_ext
import os
import ray
from os.path import exists

ray.init(runtime_env={'working_dir': ".", 'pip': "requirements.txt",
                      'env_vars': dict(os.environ),
                      'excludes': ['*.jar', '.git*/', 'jupyter/']}) if not ray.is_initialized() else True

controller = ScaledTaskController.remote()

########################
# Cache
# TODO: Use distributed backend like Ray/Dask/Gemfire
########################
cache = {}


########################
# Save to cache
########################
def save_artifact(artifact, artifact_name, distributed=True):
    try:
        save_to_cache(artifact, artifact_name)
        save_to_backend(artifact, artifact_name, distributed=distributed)
    except Exception as e:
        logging.debug(f'Could not complete execution for save_artifact - {artifact_name} - error occurred: ',
                      exc_info=True)


########################
# Save to cache
########################
def load_artifact(artifact_name, distributed=True, can_cache=True):
    try:
        if _get_sync_status(artifact_name) and can_cache:
            artifact = load_from_cache(artifact_name)
        else:
            artifact = load_from_backend(artifact_name, distributed=distributed)
        return artifact
    except Exception as e:
        logging.debug(f'Could not complete execution for load_artifact - {artifact_name} - error occurred: ',
                      exc_info=True)


########################
# Save artifact
########################
def save_to_backend(artifact, artifact_name, distributed=True):
    try:
        logging.debug(f"saving {artifact_name} to backend...{utils_ext.get_root_run_id()}")

        with open(f"/parent/app/artifacts/{artifact_name}", "wb") as artifact_handle:
            joblib.dump(artifact, artifact_handle)

        if distributed:
            controller.log_artifact.remote(utils_ext.get_root_run_id(), artifact, f"{artifact_name}")
        else:
            utils_ext.mlflow_log_artifact(utils_ext.get_root_run_id(), artifact, f"{artifact_name}")
        _set_out_of_sync(artifact_name)

        logging.debug(f"{artifact_name} saved to backend.")
    except Exception as e:
        logging.debug(f'Could not complete execution for saving {artifact_name} to backend - error occurred: ',
                      exc_info=True)


########################
# Load artifact
########################
def load_from_backend(artifact_name, distributed=True):
    artifact = None
    try:
        run_id = utils_ext.get_root_run_id()
        logging.debug(f"Loading {artifact_name} from backend with run id {run_id}...")
        if run_id:
            if distributed:
                result = controller.load_artifact.remote(run_id,
                                                         artifact_name,
                                                         artifact_uri=f"runs:/{run_id}/{artifact_name}",
                                                         dst_path="/parent/app/artifacts")
                artifact = ray.get(result)
                if artifact is not None:
                    with open(f"/parent/app/artifacts/{artifact_name}", "wb") as artifact_handle:
                        joblib.dump(artifact, artifact_handle)
            else:
                artifact = utils_ext.mlflow_load_artifact(run_id,
                                                      artifact_name,
                                                      artifact_uri=f"runs:/{run_id}/{artifact_name}",
                                                      dst_path="/parent/app/artifacts")
            _set_in_sync(artifact_name)
            logging.debug(f"{artifact_name} loaded from backend.")
    except Exception as e:
        logging.debug(f'Could not complete execution for loading {artifact_name} - error occurred: ', exc_info=True)
    finally:
        return artifact


########################
# Save model
########################
def save_model(model, model_name, flavor='sklearn', artifact_path=None):
    try:
        run_id = utils_ext.get_root_run_id()
        controller.log_model.remote(run_id,
                                    model,
                                    flavor,
                                    artifact_path=artifact_path or flavor,
                                    registered_model_name=model_name,
                                    await_registration_for=None)
    except Exception as e:
        logging.info(f'Could not complete execution for save_model - {model_name}- error occurred: ', exc_info=True)


########################
# Load model
########################
def load_model(model_name, flavor='sklearn', stage='None'):
    try:
        run_id = utils_ext.get_root_run_id()
        model_uri = f"models:/{model_name}/{stage}"
        result = controller.load_model.remote(run_id,
                                              flavor,
                                              model_uri=model_uri,
                                              dst_path="/parent/app/artifacts")
        if result is not None:
            model = ray.get(result)
        else:
            model = load_from_cache(model_name)
        return model
    except Exception as e:
        logging.info(f'Could not complete execution for load_model - {model_name}- error occurred: ', exc_info=True)


########################
# Save offset
########################
def save_offset(offset, offset_name):
    save_artifact(offset, f'{offset_name}_offset')


########################
# Load offset
########################
def load_offset(offset_name, distributed=True):
    return load_artifact(f'{offset_name}_offset', distributed=distributed)


########################
# Log metric
########################
def log_metric(metric, metric_name, distributed=True):
    run_id = utils_ext.get_root_run_id()
    logging.info(f"Logging metric {metric_name} from backend with run id {run_id}...")
    if distributed:
        controller.log_metric.remote(run_id, key=metric_name, value=metric)
    else:
        utils_ext.mlflow_log_metric(run_id, key=metric_name, value=metric)


########################
# Save to cache
########################
def save_to_cache(artifact, artifact_name):
    cache[artifact_name] = artifact
    try:
        artifact_path = f"/parent/app/artifacts/{artifact_name}"
        with open(artifact_path, "wb") as artifact_handle:
            joblib.dump(artifact, artifact_handle)
    except Exception as e:
        logging.debug(f'Could not save {artifact_name} to cache at {artifact_path} - error occurred: ', exc_info=True)
    # st.session_state[artifact_name] = artifact


########################
# Load offset
# TODO: Use distributed backend like Ray/Dask/Gemfire
########################
def load_from_cache(artifact_name):
    # return st.session_state[artifact_name]
    # return cache.get(artifact_name)
    artifact = None
    try:
        artifact_path = f"/parent/app/artifacts/{artifact_name}"
        if exists(artifact_path):
            with open(artifact_path, "rb") as artifact_handle:
                artifact = joblib.load(artifact_handle)
                logging.debug(f"Artifact {artifact_path} loaded from cache.")
    except Exception as e:
        logging.debug(f'Could not load artifact {artifact_name} from cache - error occurred: ', exc_info=True)
    finally:
        return artifact


########################
# Set in_sync flag=True for this artifact in the cache
# (used to implement dirty read flag for cache)
########################
def _set_in_sync(artifact_name):
    cache[f"{artifact_name}_in_sync_flag"] = True


########################
# Set out_of_sync flag=False for this artifact in the cache
# (used to implement dirty read flag for cache)
########################
def _set_out_of_sync(artifact_name):
    cache[f"{artifact_name}_in_sync_flag"] = False


########################
# Set in-sync status for this artifact in the cache
# (used to implement dirty read flag for cache)
########################
def _get_sync_status(artifact_name):
    return cache.get(f"{artifact_name}_in_sync_flag")
