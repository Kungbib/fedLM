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

    outfile_name = 'electra_seed'
    create_graph(settings)
    weights = get_weights_from_model(settings)
    helper = PytorchHelper()
    helper.save_model(weights, outfile_name)
