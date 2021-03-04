import sys
import json
import yaml

from .client.src.clienthelper_tfestimator import (create_graph,
                                                  get_weights_from_model
                                                  )
from fedn.utils.pytorchhelper import PytorchHelper

if __name__ == '__main__':
    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)
    with open(sys.argv[1]) as fh:
        hparams = json.load(fh)
    data_dir = settings["data"]
    model_name = settings["model_name"]
    # data_dir = 'data'
    # model_name = 'electra_small_owt'
    with open(sys.argv[1]) as fh:
        hparams = json.load(fh)
    data_dir = hparams["data_dir"]
    model_name = hparams["model_name"]
    outfile_name = 'electra_seed'
    create_graph(data_dir, model_name)
    weights = get_weights_from_model(data_dir, model_name)
    helper = PytorchHelper()
    helper.save_model(weights, outfile_name)
