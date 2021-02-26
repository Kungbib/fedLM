from __future__ import print_function
import tensorflow as tf
import yaml
import sys
sys.path.append('scripts/electra_script')
from run_pretraining import train_or_eval
import configure_pretraining
from fedn.utils.pytorchhelper import PytorchHelper
from scripts.clienthelper_tfestimator import load_weights_to_model, get_weights_from_model, get_global_step

def train(data_dir, model_name, settings):
    print("-- RUNNING TRAINING --", flush=True)
    import os
    arr = os.listdir(data_dir)
    print(arr)

    global_step = get_global_step(data_dir, model_name)

    hparams = {
        "num_train_steps": global_step + settings['num_train_steps'],
        "save_checkpoints_steps": 10,
        "train_batch_size": settings['train_batch_size']
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
    # TODO: define model_name and data_dir in settings
    model_name = 'electra_small_owt'
    data_dir = '/app/data'

    helper = PytorchHelper()
    weights = helper.load_model(sys.argv[1])

    load_weights_to_model(weights, data_dir, model_name)
    train(data_dir, model_name, settings)
    weights = get_weights_from_model(data_dir, model_name)
    ret = helper.save_model(weights, sys.argv[2])
    print("saved as: ", ret)
