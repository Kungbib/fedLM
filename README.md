# fedLM

A federated LM

## Electra

### Setup

Clone from original source

```bash
git clone git@github.com:google-research/electra.git
```

Then use `docker-compose` files in `test` folder to run the preprocessing and
other necessary steps.
First move them into the `electra` folder, then build and run.

Build the docker container:

```bash
docker-compose -f docker-compose.yaml build
```

and run it interactively

```bash
docker-compose run -u $(id -u):$(id -g) --rm electra bash
```

### Pretraining

```bash
python3 build_pretraining_dataset.py --corpus-dir ../data/${lang} --vocab-file ../data/vocab.${lang}.txt --output-dir ./data/ --max-seq-length 128 --num-processes 15 --blanks-separate-docs True --do-lower-case
```


