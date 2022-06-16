import itertools

from simple_slurm import Slurm
import platform
import subprocess

is_windows = platform.system() == 'Windows'
#%%
params_to_vary = {
    "experiment_name": [
        "SNLSanitycheck3",
    ],
    "seed": [x for x in range(5)],
}

keys = sorted(params_to_vary.keys())

vals = [params_to_vary[k] for k in keys]

param_combinations = list(itertools.product(*vals))

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
    cur_function = "python train_lstm_SNL.py "

    for j, key in enumerate(keys):
        cur_function += "--" + key + " " + str(param_combinations[i][j]) + " "

    if is_windows:
        subprocess.call(cur_function, shell=True)
        # print(cur_function)
    else:
        slurm.sbatch(cur_function)
