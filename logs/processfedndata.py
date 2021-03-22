import json
import matplotlib.pyplot as plt
from collections import defaultdict


def read_status(fname):
    with open(fname) as fh:
        json_data = json.load(fh)
        results = []
        for entry in filter(lambda x: x["type"] == "MODEL_VALIDATION", json_data):
            entry = json.loads(entry["data"])
            results.append((json.loads(entry["data"]), entry["modelId"]))
    return results


def plot_status(results, metric="loss"):
    xs = defaultdict(list)
    for x, name in results:
        xs[name].append(x[metric])
    plt.plot(list(xs.values()), "-")
    plt.show()


if __name__ == "__main__":
    import sys
    data = read_status(sys.argv[1])
    plot_status(data, "loss")
