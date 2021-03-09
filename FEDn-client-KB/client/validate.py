import tensorflow.compat.v1 as tf
import yaml
import sys
import json

sys.path.append('src/electra')
from run_pretraining import train_or_eval
import configure_pretraining
from fedn.utils.pytorchhelper import PytorchHelper
from src.clienthelper_tfestimator import load_weights_to_model


def validate(settings):
    print("-- RUNNING VALIDATE --", flush=True)

    with open(settings["hparams"]) as fh:
        hparams = json.load(fh)
    # overwrite hparams with settings
    for setting in settings:
        if setting in hparams:
            hparams[setting] = settings[setting]
    data_dir = settings["data_dir"]
    model_name = settings["model_name"]

    conf = configure_pretraining.PretrainingConfig(
        model_name, data_dir, **hparams)

    conf.do_train = False
    conf.do_eval = True

    tf.logging.set_verbosity(tf.logging.ERROR)
    ret = train_or_eval(conf, device=settings["val_device"])
    report = {
        "classification_report": 'unevaluated',
        "loss": float(ret['loss']),
        "accuracy": float(ret['disc_accuracy'])
    }
    return report


def _make_report(settings):
    report = validate(settings)
    print("report: ", report)
    return report


def make_report(in_arg):
    with open('/app/settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)
    helper = PytorchHelper()
    weights = helper.load_model(in_arg)

    load_weights_to_model(weights, settings)
    # report = validate(settings)
    # print("report: ", report)
    # return report
    return _make_report(settings)


def write_report(report, out_arg):
    print("sys.argv[2]: ", out_arg)
    with open(out_arg, "w") as fh:
        fh.write(json.dumps(report))


def read_report(out_arg):
    with open(out_arg, "r") as fh:
        report = json.loads(fh)
    return report


if __name__ == '__main__':

    # with open('/app/settings.yaml', 'r') as fh:
    #     try:
    #         settings = dict(yaml.safe_load(fh))
    #     except yaml.YAMLError as e:
    #         raise(e)

    # helper = PytorchHelper()
    # weights = helper.load_model(sys.argv[1])

    # load_weights_to_model(weights, settings)
    # report = validate(settings)
    # print("report: ", report)
    # print("sys.argv[2]: ", sys.argv[2])
    # with open(sys.argv[2], "w") as fh:
    #     fh.write(json.dumps(report))

    # report = make_report(sys.argv[1])
    report = read_report("/app/last_validate.json")
    write_report(report, sys.argv[2])
