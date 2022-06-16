import itertools
import os

from simple_slurm import Slurm
import platform

is_windows = platform.system() == 'Windows'
params_to_vary = {
    "experiment_name": [
        "NumBatteries",
    ],
    "train_percentage": [0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1],
    "seed": [x for x in range(25)],
}

keys = sorted(params_to_vary.keys())

vals = [params_to_vary[k] for k in keys]

param_combinations = list(itertools.product(*vals))  # list of tuples
print(len(param_combinations))
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

    cur_function = "python train_lstm.py "

    for j, key in enumerate(keys):

        cur_function += "--" + key + " " + str(param_combinations[i][j]) + " "

    if is_windows:
        subprocess.call(cur_function, shell=True)
        # print(cur_function)
    else:
        slurm.sbatch(cur_function)
