import json


def json_to_txt(fn_in: str, fn_out: str) -> None:
    with open(fn_in) as fh_in, open(fn_out, "w") as fh_out:
        for line in fh_in:
            jline = json.loads(line)
            if jline["text"] != "":
                print(jline["text"].strip(), file=fh_out, end="\n\n")


if __name__ == "__main__":
    import sys
    json_to_txt(sys.argv[1], sys.argv[2])
