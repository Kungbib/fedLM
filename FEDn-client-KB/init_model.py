import sys
sys.path.append('client/scripts/electra_script')

from client.scripts.clienthelper_tfestimator import create_graph, get_weights_from_model
from fedn.utils.pytorchhelper import PytorchHelper

if __name__ == '__main__':
    data_dir = 'data'
    model_name = 'electra_small_owt'
    outfile_name = 'electra_seed'
    create_graph(data_dir, model_name)
    weights = get_weights_from_model(data_dir, model_name)
    helper = PytorchHelper()
    helper.save_model(weights, outfile_name)


