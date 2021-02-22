# /bin/env python3
# coding: utf-8

import sys
import stanza
from typing import List
from tqdm import tqdm


lang = sys.argv[2]  # State the segmenter model (no or nn or da or sv)

nlp = stanza.Pipeline(lang, processors="tokenize")

stack: List[str] = []

with open(sys.argv[1]) as fh_in:
    file_size = 0
    for line in fh_in:
        file_size += 1

with open(sys.argv[1]) as fh_in:
    for line in tqdm(fh_in, total=file_size):
        if line == "\n":
            texts = "\n".join(stack)
            try:
                doc = nlp(texts)
                for sentence in doc.sentences:
                    print(sentence.text)
            except RuntimeError:
                print(len(stack), file=sys.stderr)
                b = int(len(stack) / 2)
                stack1 = stack[:b]
                stack2 = stack[b:]
                texts1 = "\n".join(stack1)
                texts2 = "\n".join(stack2)
                doc1 = nlp(texts1)
                doc2 = nlp(texts2)
                for sentence in doc1.sentences + doc2.sentences:
                    print(sentence.text)
            stack = []
            print("")
            continue
        stack.append(line.strip())
