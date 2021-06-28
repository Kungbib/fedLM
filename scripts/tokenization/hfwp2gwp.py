import json

UNUSED_FORMAT = '[unused{}]'


def hugface_to_google(fn_in: str, fn_out: str, unused_count: int) -> None:
    with open(fn_in) as fh:
        hf_vocab = list(json.load(fh)["model"]["vocab"].keys())

    unused = [UNUSED_FORMAT.format(i) for i in range(unused_count)]
    go_vocab = [hf_vocab[0]] + unused + hf_vocab[1:]
    with open(fn_out, "w") as fh:
        for v in go_vocab:
            print(v, file=fh)


if __name__ == "__main__":
    import sys
    hugface_to_google(sys.argv[1], sys.argv[2], 1000)
