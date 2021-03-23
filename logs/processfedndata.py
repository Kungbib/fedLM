import json
import matplotlib.pyplot as plt
from collections import defaultdict
import datetime
import numpy as np


def read_status(fname: str):
    with open(fname) as fh:
        json_data = json.load(fh)
        results = []
        for entry in filter(lambda x: x["type"] == "MODEL_VALIDATION",
                            json_data):
            entry = json.loads(entry["data"])
            results.append((json.loads(entry["data"]),
                            entry["modelId"],
                            entry["timestamp"]))
    return results


def plot_status(results, steps_per_round, metric="loss", time=False):
    ys = defaultdict(list)
    for y, name, timestamp in results:
        ys[name].append(y[metric])
    if time:
        xs = {}
        start = None
        for i, (_, name, timestamp) in enumerate(results):
            time = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
            if start is None:
                start = time
            time_passed = time - start
            xs[name] = time_passed.total_seconds()
        xs = xs.values()
    else:
        xs = np.arange(0, len(ys) * steps_per_round, steps_per_round)
    plt.plot(xs, list(ys.values()), "-")


def print_status(results, steps_per_round, metric="loss"):
    xs = defaultdict(list)
    for x, name, timestamp in results:
        time = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
        xs[name].append((x[metric], time))
    start = None
    for i, x in enumerate(xs):
        (v1, t1), (v2, t2) = xs[x]
        if start is None:
            start = t1
        time_passed = t1 - start
        print(i * steps_per_round, time_passed, v1, v2, (v1 + v2) / 2)


if __name__ == "__main__":
    import sys
    data1 = read_status(sys.argv[1])
    print_status(data1, int(sys.argv[2]), "loss")
    plot_status(data1, int(sys.argv[2]), "loss", time=True)
    data2 = read_status(sys.argv[3])
    plot_status(data2, int(sys.argv[4]), "loss", time=True)
    plt.show()
