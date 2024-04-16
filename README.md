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
