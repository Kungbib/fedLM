import re
from typing import Dict, Set, Tuple


def read_vocab(fn: str) -> Dict[str, int]:
    vocab = {}
    with open(fn) as fh:
        for i, line in enumerate(fh):
            vocab[line.strip()] = i
    return vocab


def filter_special(vocab: Dict[str, int]) -> Set[str]:
    words = set()
    for word in vocab:
        if not re.match(r"\[\S+\]", word):
            words.add(word)
    return words


def compare_2(v1: Set[str], v2: Set[str]
              ) -> Tuple[Tuple[int, int, int], Set[str]]:
    print(f"len(v1): {len(v1)}", end="\t")
    print(f"len(v2): {len(v2)}", end="\t")
    lsv = len(v1.intersection(v2))
    print(f"shared: {lsv}", end="\t")
    v_no_digit = set(filter(lambda x: not re.match(r"##\d+", x), v1.intersection(v2)))
    v_no_weird = set(filter(lambda x: re.match(r"##[a-zäöåøæüßA-ZÄÖÅÆØÜ]", x), v_no_digit))
    lvnd = len(v_no_digit)
    print(f"digits: {lsv - lvnd}", end="\t")
    lvnw = len(v_no_weird)
    print(f"weird: {lvnd - lvnw}", end="\t")
    print()
    return (lsv, lvnd, lvnw), v_no_weird


if __name__ == "__main__":
    import sys
    v = compare_2(filter_special(read_vocab(sys.argv[1])),
                  filter_special(read_vocab(sys.argv[2]))
                  )
    # print("\n".join(v))
