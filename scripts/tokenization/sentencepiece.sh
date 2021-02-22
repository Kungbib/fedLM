#!/bin/bash

out=$1
shift
size=$1
shift
files=$@

echo "Output file: $out"
echo "Vocabulary Size: $size"
echo "Corpus files: $files"

python3 spiece_tokenizer.py $out $size $files
python3 sent2wordpiece.py $out.json -o $out.txt
