# Download and Prepare the Data

## OSCAR

Run the python script: 

```python
python get_data.py
```

This will download the OSCAR data for no, nn, da, sv (uncomment da if
necessary) and place the files into a new `data/oscar` folder in the main
directory.

Downloading via `datasets` has the "benefit" of caching the data in
`.cache/huggingface/datasets`.
This may result into disk problems and can be removed if necessary.

## Wikipedia Dumps

For Wikipedia we first download the dumps, process them with **wikiextractor**
and then extract the text from the `json` files to save the data into
`data/wiki`.

The version for **wikiextractor** is not in **pip** and has to be _cloned_ and
installed from **github** in a reasonable directory.

```bash
git clone https://github.com/attardi/wikiextractor.git 
cd wikiextractor
python setup.py install
```

Then:

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
