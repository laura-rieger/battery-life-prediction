import itertools
from simple_slurm import Slurm
import subprocess
import platform

is_windows = platform.system() == 'Windows'
#%%
params_to_vary = {
    "experiment_name": [
        "DNN",
    ],
    "dropout": [
        0.0,
    ],
    "hidden_size": [128, 256, 512],
    "sequence_length": [
        40,
        100,
    ],
    "train_percentage": [
        1,
    ],
    "start": [
        10,
    ],
    "seed": [x for x in range(25)],
    "num_epochs": [
        1000,
    ],
    "bootstrap": [
        1,
    ],  # train only on part of the dataset, f.e. 50% -> /5
}

keys = sorted(params_to_vary.keys())

vals = [params_to_vary[k] for k in keys]

param_combinations = list(itertools.product(*vals))  # list of tuples

for i in range(len(param_combinations)):
    slurm = Slurm(
        mail_type="FAIL",
        partition="sm3090",
        N=1,
        n=8,
        time="0-00:15:00",
        mem="10G",
        gres="gpu:RTX3090:1",
    )
    cur_function = "python train_dnn.py "

    for j, key in enumerate(keys):

        cur_function += "--" + key + " " + str(param_combinations[i][j]) + " "
    if is_windows:
        subprocess.call(cur_function, shell=True)

    else:
        slurm.sbatch(cur_function)
