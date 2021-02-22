import os
from datasets import load_dataset

# OSCAR
# OPUS 100
# Wiki?


def oscar_to_text(lang: str, out_fn: str) -> None:
    try:
        ds = load_dataset("oscar", f"unshuffled_deduplicated_{lang}",
                          script_version="master")
    except ValueError:
        print(f"The oscar dataset with {lang} does not exist")
        return

    # for ttv in ds:  # there is only train
    with open(f"{out_fn}.{lang}", "w") as fh:
        for line in ds["train"]:
            print(line["text"], file=fh, end="\n\n")
    return


def opus100_to_text(lang: str, out_fn: str) -> None:
    try:
        ds = load_dataset("opus100", f"en-{lang}")
    except ValueError:
        print(f"The opus100 dataset with {lang} does not exist")
        try:
            ds = load_dataset("opus100", f"{lang}-en")
        except ValueError:
            print(f"The opus100 dataset with {lang} does not exist")
            return

    for ttv in ds:
        with open(f"{out_fn}.{ttv}.{lang}", "w") as fh:
            for line in ds[ttv]:
                print(line["translation"][lang], file=fh, end="\n\n")
    return


if __name__ == "__main__":
    langs = ["sv", "no", "nb", "nn", "da"]
    corpora = ["oscar", "opus100"]  # cc100 open_subtitles
    os.makedirs("../data/oscar", exist_ok=True)
    # os.makedirs("../data/opus100", exist_ok=True)
    for lang in langs:
        oscar_to_text(lang, "../data/oscar/oscar")
        # opus100_to_text(lang, "../data/opus100/opus100")
