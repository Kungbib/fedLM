import yaml
import sys
from train import train
import validate as vldt
from fedn.utils.pytorchhelper import PytorchHelper
from src.clienthelper_tfestimator import (load_weights_to_model,
                                          get_weights_from_model,
                                          )


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
    for i in range(10_001):
        weights = helper.load_model(prev_model)

        load_weights_to_model(weights, settings)
        train(settings)
        weights = get_weights_from_model(settings)
        # validate
        report = vldt._make_report(settings)
        vldt.write_report(report, "/app/last_validate.json")

        prev_model = f"/app/weights/electra_weigths.{i}.npz"
        ret = helper.save_model(weights, prev_model)
        print("saved as: ", ret)
