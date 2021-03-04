from __future__ import print_function
import tensorflow as tf
import yaml
import sys
import json
sys.path.append('scripts/electra')
from run_pretraining import train_or_eval
import configure_pretraining
from fedn.utils.pytorchhelper import PytorchHelper
from scripts.clienthelper_tfestimator import load_weights_to_model, get_weights_from_model

def validate(data_dir, model_name, settings):
    print("-- RUNNING VALIDATE --", flush=True)

    # hparams = {"num_eval_steps": 1}#, "train_batch_size": 128}
    with open(settings.hparams) as fh:
        hparams = json.load(fh)
    
    # overwrite hparams with settings
    for setting in settings:
        if setting in hparams:
            hparams[setting] = settings[setting]
    conf = configure_pretraining.PretrainingConfig(
        model_name, data_dir, **hparams)

    conf.do_train = False
    conf.do_eval = True

    tf.logging.set_verbosity(tf.logging.ERROR)
    ret = train_or_eval(conf, device='cpu')
    report = {
        "classification_report": 'unevaluated',
        "loss": float(ret['loss']),
        "accuracy": float(ret['disc_accuracy'])
    }
    return report

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)
    # TODO: define model_name and data_dir in settings
    model_name = settings["model_name"] 
    data_dir = settings["data"] 

    helper = PytorchHelper()
    weights = helper.load_model(sys.argv[1])

    load_weights_to_model(weights, data_dir, model_name)
    report = validate(data_dir, model_name, settings)
    print("report: ", report)
    print("sys.argv[2]: ", sys.argv[2])
    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))
