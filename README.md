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

Create shards and put everything into the TF format.

```bash
python3 build_pretraining_dataset.py --corpus-dir ../data/${lang} --vocab-file ../data/vocab.${lang}.txt --output-dir ./data/ --max-seq-length 128 --num-processes 15 --blanks-separate-docs True --do-lower-case
```

Then run the pre-training:

```bash
python3 run_pretraining.py --data-dir data/ --model-name electra_small_$lang --hparams '{"debug": false, "do_train": true, "do_eval": false, "vocab_size": 31000, "vocab_file": "vocab.$lang.txt"}'
```

### Robin Forgets

- kb-labb-1 Danish
- kb-labb-2 Norwegian
- kb-labb-3 Swedish

#### Checkpoints @

- 50% - 500k
- 25% - 250k
- 12.5% - 125k
- 6.25% - 62.5k

#### Time Estimates

Strongly depends on GPU (yay RTX 3090).

RTX 3090 : 13.5k / h
RTX 2080 :  7.8k / h
RTX 2060 :  ?.?k / h

