from __future__ import print_function
import tensorflow as tf
import yaml
import sys
import json
sys.path.append('scripts/electra')
from run_pretraining import train_or_eval
import configure_pretraining
from fedn.utils.pytorchhelper import PytorchHelper
from scripts.clienthelper_tfestimator import load_weights_to_model, get_weights_from_model, get_global_step

def train(data_dir, model_name, settings, client_settings):
    print("-- RUNNING TRAINING --", flush=True)

    global_step = get_global_step(data_dir, model_name)

    with open(settings.hparams) as fh:
        hparams = json.load(fh)
    # overwrite hparams with settings
    for setting in settings:
        if setting in hparams:
            hparams[setting] = settings[setting]
    try:
        train_batch_size = client_settings['train_batch_size']
    except:
        train_batch_size = settings['train_batch_size']

    try:
        num_train_steps = client_settings['num_train_steps']
    except:
        num_train_steps = settings['num_train_steps']

    print("global_step: ", global_step)
    hparams = {
        "num_train_steps": global_step + num_train_steps,
        # "save_checkpoints_steps": 10,
        "train_batch_size": train_batch_size
    }

    tf.logging.set_verbosity(tf.logging.ERROR)
    train_or_eval(configure_pretraining.PretrainingConfig(
        model_name, data_dir, **hparams))
    print("Training done!")


if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    client_settings = {}
    with open('/app/client_settings.yaml', 'r') as fh:
        try:
            client_settings = dict(yaml.safe_load(fh))
            print("client_settings: ", client_settings)
        except yaml.YAMLError as e:
            raise(e)
    # TODO: define model_name and data_dir in settings
    model_name = settings["model_name"]
    data_dir = settings["data"]

    helper = PytorchHelper()
    weights = helper.load_model(sys.argv[1])

    load_weights_to_model(weights, data_dir, model_name)
    train(data_dir, model_name, settings, client_settings)
    weights = get_weights_from_model(data_dir, model_name)
    ret = helper.save_model(weights, sys.argv[2])
    print("saved as: ", ret)
