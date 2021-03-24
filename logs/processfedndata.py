import json
import matplotlib.pyplot as plt
from collections import defaultdict
import datetime
import numpy as np


def smoothen(xs, ys, size=10):
    _xs = []
    _ys = []
    _x = 0
    _y1 = 0
    _y2 = 0
    for i, (x, (y1, y2)) in enumerate(zip(xs, ys)):
        _x += x
        _y1 += y1
        _y2 += y2
        if (i + 1) % size == 0:
            _xs.append(_x / size)
            _ys.append((_y1 / size, _y2 / size))
            _x = 0
            _y1 = 0
            _y2 = 0
    return _xs, _ys


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


def plot_status(results,
                steps_per_round,
                metric="loss",
                time=False,
                smooth=False):
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
    ys = list(ys.values())
    if smooth:
        xs, ys = smoothen(xs, ys)
    plt.plot(xs, ys, "-")


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
    time_spent = False
    smooth = True
    plot_status(data1, int(sys.argv[2]), "loss", time=time_spent, smooth=smooth)
    data2 = read_status(sys.argv[3])
    plot_status(data2, int(sys.argv[4]), "loss", time=time_spent, smooth=False)
    plt.show()
