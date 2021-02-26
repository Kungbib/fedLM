#! /bin/env python3

import sys

from tokenizers import BertWordPieceTokenizer

if __name__ == "__main__":

    tokenizer = BertWordPieceTokenizer(lowercase=True)

    tokenizer.train(sys.argv[3:], vocab_size=int(sys.argv[2]))
    tokenizer.save(sys.argv[1] + ".json")
