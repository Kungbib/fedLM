# fedLM
A federated LM

## To initiate and start the Electra client in FEDn


### Pre steps
Create a `data` folder inside `FEDn-client-KB`:
```
mkdir data
```
put the `vocab.txt` and the `pretrain_tfrecords` inside the data folder.

build the compute package inside `FEDn-client-KB`:
```
mkdir package
tar -X ../.gitignore --exclude "*__pycache__*" --exclude "*.swp" -czvf package/electra.tar.gz client/
```
build the seed model inside `FEDn-client-KB`:
```

```

### Start Minio/Mongo, Reducer and Combiner
Clone the 
[FEDn](https://github.com/scaleoutsystems/fedn/tree/develop) repository and 
follow the instructions.\
Before starting the Reducer make sure the helper is set to _pytorch_ in `config/settings_reducer.yaml` (line 5) to have _pytorch_.
```
  helper: pytorch
```


### Add client to FEDn
