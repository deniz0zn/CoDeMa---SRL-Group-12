# CoDeMa---SRL-Group-12

How to set up below :)

we have the following tree structure of the project:

```text
(CoDeMa---SRL-Group-12) dozen@DeniZenbook:~/SRL/CoDeMa---SRL-Group-12$ tree .
.
├── README.md
|
├── configs
│   ├── dataset.yaml            --- where we set up data-related configuration
│   ├── models                  --- each of the model init/configs go here, i placed some examples inside
│   │   ├── cdt.yaml
│   │   ├── cdm.yaml
│   │   └── cdt_small.yaml
│   └── train.yaml              --- training hyperparams
|
├── models
│   └── cdt.py                  --- our wrapper from OSRL
│   └── cdm.py                  --- mamba extension on CDT baseline
|
├── pyproject.toml
├── train.py                    --- main training calls, we can directly pass a model config to train it
├── utils.py                    --- logging and wandB related stuff
└── uv.lock
```


To set up the project:

1. Ensure Linux/WSL, the training envs/gym won't work with Windows
2. Ensure uv

Then,
1. Clone the project and `cd` into it,
2. Run:
```shell
uv venv --python /usr/bin/python3.10
uv sync
source .venv/bin/activate
```

The `pyproject.toml` is set in such way that it automatically sets up the additional libraries from git.

3. Run a model:
```shell
uv run python train.py --model configs/models/cdt.yaml 
```
## Logging
We are now using wandB because it is nice way of monitoring the training and evaluation metrics which can be observed simultaneously by all group members
https://forge.coreweave.com/wandb/dozen/constrained-decision-models/workspace

When running for the first time wandB will ask you to create an account and paste the API key (if there is an error with the key length run `uv sync`)

We now log every training evaluation a model instances/objects under `runs/`


---


Todos/nice to have(s):
- a vis utility, both for training and environment
- Find a way to visualize environment (i.e.,car driving)