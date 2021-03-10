import matplotlib.pyplot as plt
import json


def plot(fn: str) -> None:
    with open(fn) as fh:
        fc = json.load(fh)
    tt, ss, ll = zip(*fc)
    plt.plot(ss, ll)
    plt.show()


if __name__ == "__main__":
    import sys
    plot(sys.argv[1])
