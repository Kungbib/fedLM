# Download and Prepare the Data

## OSCAR

Run the python script: 

```python
python get_data.py
```

This will download the OSCAR data for no, nn, da, sv (uncomment da if
necessary) and place the files into a new `data/oscar` folder in the main
directory.

## Wikipedia Dumps

For Wikipedia we first download the dumps, process them with **wikiextractor**
and then extract the text from the `json` files to save the data into
`data/wiki`.

```bash
bash get_wiki.sh
```

## Sentence Segmentation

Sentence segmentation is done with the sentence splitter coming with the 
venerable **mosesdecoder**.
The `perl` script is called on all the data with

```bash
bash sentence_segment_all.sh
```

## Vocabulary Generation

The vocabulary is created using **sentencepiece** with scripts in the 
`tokenization` folder.

```bash
bash sentencepiece.sh $vocab $size @input_files
```
