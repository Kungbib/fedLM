import tensorflow as tf
import yaml
import sys
import json
# sys.path.append('src/electra')
from .src.electra.run_pretraining import train_or_eval
from .src.electra import configure_pretraining
from fedn.utils.pytorchhelper import PytorchHelper
from .src.clienthelper_tfestimator import (load_weights_to_model,
                                           get_weights_from_model,
                                           get_global_step
                                           )


def train(data_dir, model_name, settings):
    print("-- RUNNING TRAINING --", flush=True)

    global_step = get_global_step(data_dir, model_name)

    with open(settings.hparams) as fh:
        hparams = json.load(fh)
    # overwrite hparams with settings
    for setting in settings:
        if setting in hparams:
            hparams[setting] = settings[setting]

    print("global_step: ", global_step)
    hparams["num_train_steps"] = global_step + settings["num_train_steps"]

    tf.logging.set_verbosity(tf.logging.ERROR)
    train_or_eval(configure_pretraining.PretrainingConfig(
        model_name, data_dir, **hparams))
    print("Training done!")


if __name__ == '__main__':

    settings = {}
    with open('/app/settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
            print("settings: ", settings)
        except yaml.YAMLError as e:
            raise(e)
    model_name = settings["model_name"]
    data_dir = settings["data"]
    hparams_fn = settings["hparams"]

    helper = PytorchHelper()
    weights = helper.load_model(sys.argv[1])

    load_weights_to_model(weights, data_dir, model_name)
    train(data_dir, model_name, settings)
    weights = get_weights_from_model(data_dir, model_name, hparams_fn)
    ret = helper.save_model(weights, sys.argv[2])
    print("saved as: ", ret)
