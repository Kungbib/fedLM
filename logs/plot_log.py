import matplotlib.pyplot as plt
import json

def plot(fn: str) -> None:
    with open(fn) as fh:
        fc = json.load(fh)
    ll = [l[2] for l in fc]
    plt.plot(ll)
    plt.show()


if __name__ == "__main__":
    import sys
    plot(sys.argv[1])
