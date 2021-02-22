#! /bin/env python3

import sys
from tokenizers import SentencePieceBPETokenizer, normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace

tokenizer = SentencePieceBPETokenizer()
tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()

tokenizer.train(sys.argv[3:], vocab_size=int(sys.argv[2]))

tokenizer.save(sys.argv[1] + ".json")
