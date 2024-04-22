# WIMU - Symbolic music model hub

## Environment setup
Hub uses Anaconda for package management, which can be installed from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html). 

The following command can be used to create the environment from file:
```sh
conda env create --name wimu --file environment.yml
```

This environment may then be activated with:
```sh
conda activate wimu
```

If the environment is changed by the user, then it should be updated with:
```sh
conda env export --name wimu | grep -v "prefix: " > environment.yml
```

## Training models from scratch
To train model from scratch run the following command:
```sh
python3 -m scripts.train -p CONFIG_FILE 
```
where `CONFIG_FILE` is a path to config file of the model you want to train

To train model from checkpoint run:
```sh
python3 -m scripts.train -p CONFIG_FILE -c CKPT_PATH
```
where `CKPT_PATH` is a path to model checkpoint


## Sampling from model
To sample from a model run:
```sh
python3 -m scripts.sample -p CONFIG_FILE -c CKPT_PATH -b BS -o OUT
```
where `BS` is the number of samples you want to generate and `OUT` is a path to directory the samples should be saved to
