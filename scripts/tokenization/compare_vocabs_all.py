import itertools as it
import compare_vocabs as cv
from typing import Dict, Tuple
from colour import colour

RED = colour.RED
BLUE = colour.BLUE
BOLD = colour.BOLD
END = colour.END


def compare(l1, f1, s1, l2, f2, s2, cache):
    fn_1 = f"spieces/vocab.{l1}.{f1}.{s1}.txt"
    fn_2 = f"spieces/vocab.{l2}.{f2}.{s2}.txt"
    if (fn_1, fn_2) not in cache:
        print(f"{BOLD}{RED}{fn_1}{END} || {BOLD}{BLUE}{fn_2}{END}")
        try:
            cnts, _ = cv.compare_2(cv.filter_special(cv.read_vocab(fn_1)),
                                   cv.filter_special(cv.read_vocab(fn_2))
                                   )
            cache[(fn_1, fn_2)] = cnts
        except FileNotFoundError:
            cache[(fn_1, fn_2)] = (-1, -1, -1)
        print()
    return cache


if __name__ == "__main__":

    langs = ["sv", "no", "da", "sv+no+da", "sv+no"]
    files = ["wiki", "oscar", "oscar+wiki"]
    sizes = ["30000", "50000"]

    cache: Dict[Tuple[str, str], Tuple[int, int, int]] = {}

    for (l1, l2) in it.combinations(langs, 2):
        for (f1, f2) in it.combinations(files, 2):
            for (s1, s2) in it.combinations(sizes, 2):
                # same language
                compare(l1, f1, s1, l1, f2, s1, cache)
                compare(l1, f1, s2, l1, f2, s2, cache)
                compare(l1, f1, s1, l1, f2, s2, cache)
                compare(l1, f1, s1, l1, f1, s2, cache)
                compare(l1, f2, s1, l1, f2, s2, cache)

                # different language same data
                compare(l1, f1, s1, l2, f1, s1, cache)
                compare(l1, f1, s2, l2, f1, s2, cache)
                compare(l1, f1, s1, l2, f1, s2, cache)
                compare(l1, f2, s1, l2, f2, s1, cache)
                compare(l1, f2, s2, l2, f2, s2, cache)
                compare(l1, f2, s1, l2, f2, s2, cache)

                # different language different data
                compare(l1, f1, s1, l2, f2, s1, cache)
                compare(l1, f1, s2, l2, f2, s2, cache)
                compare(l1, f1, s1, l2, f2, s2, cache)
