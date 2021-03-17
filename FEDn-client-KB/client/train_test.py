import sys
import yaml
import json
import tensorflow as tf
import validate as vldt
from fedn.utils.pytorchhelper import PytorchHelper
from src.clienthelper_tfestimator import (load_weights_to_model,
                                          get_weights_from_model,
                                          )
sys.path.append('src/electra')
from run_pretraining import train_or_eval
import configure_pretraining


def train(settings, global_step):
    print("-- RUNNING TRAINING --", flush=True)

    data_dir = settings["data_dir"]
    model_name = settings["model_name"]

    with open(settings["hparams"]) as fh:
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

    helper = PytorchHelper()
    prev_model = "/app/electra_seed.npz"
    old_weights = helper.load_model(prev_model)
    load_weights_to_model(old_weights, settings)

    global_step = 100
    for i in range(10_001):
        # old_weights = helper.load_model(prev_model)

        # load_weights_to_model(old_weights, settings)
        train(settings, global_step)
        # new_weights = get_weights_from_model(settings)
        # validate
        if i % 100 == 0:
            report = vldt._make_report(settings)
            vldt.write_report(report, "/app/last_validate.json")
        global_step += 100

        # prev_model = f"/app/weights/electra_weigths.{i}.npz"
        # ret = helper.save_model(new_weights, prev_model)
        # print("saved as: ", ret)
