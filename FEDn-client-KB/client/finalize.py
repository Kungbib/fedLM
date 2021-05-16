# import tensorflow.v1.compat as tf
import tensorflow as tf
import yaml
import sys
import json
import validate as vldt

sys.path.append('src/electra')
from run_pretraining import train_or_eval
import configure_pretraining
from fedn.utils.pytorchhelper import PytorchHelper
from src.clienthelper_tfestimator import (load_weights_to_model,
                                          get_weights_from_model,
                                          get_global_step
                                          )


if __name__ == '__main__':

    helper = PytorchHelper()
    weights = helper.load_model(sys.argv[1])

    settings = {}
    settings["data_dir"] = sys.argv[2]
    settings["model_name"] = sys.argv[3]

    load_weights_to_model(weights, settings)

