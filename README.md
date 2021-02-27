# fedLM

A federated LM

## Electra

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
docker-compose run --rm electra bash
```
