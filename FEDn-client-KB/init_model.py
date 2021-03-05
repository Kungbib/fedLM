import sys
import json
import yaml

from client.src.clienthelper_tfestimator import (create_graph,
                                                 get_weights_from_model
                                                 )
from fedn.utils.pytorchhelper import PytorchHelper

if __name__ == '__main__':
    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)
    data_dir = settings["data"]
    model_name = settings["model_name"]
    print(model_name, type(model_name))
    outfile_name = 'electra_seed'
    create_graph(data_dir, model_name, settings["hparams"])
    weights = get_weights_from_model(data_dir, model_name, settings["hparams"])
    helper = PytorchHelper()
    helper.save_model(weights, outfile_name)
