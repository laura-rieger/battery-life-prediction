# In[2]:

import configparser
import os
import pickle as pkl
import sys
from os.path import join as oj

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import cycler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm, trange

import my_eval

sys.path.insert(0, "../src_lstm")
import pickle as pkl

import seaborn as sns

import models
import my_eval
import severson_data

pd.set_option("display.float_format", lambda x: "%.2f" % x)
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# In[3]:

colours = [
    "#208F90", "#8F2317", "#17608F", "#8F5F17", "#f2f3f4", "#E56399", "#DE6E4B"
]
golden_ratio = 1.618

sns.set_palette(sns.color_palette(colours))

colors = cycler("color", colours)
plt.rc(
    "axes",
    facecolor="#FFFFFF",
    edgecolor="#000000",
    axisbelow=True,
    grid=True,
    prop_cycle=colors,
)

# In[4]:

config = configparser.ConfigParser()
config.read("../config.ini")
result_path = config["PATHS"]["result_path"]

# In[5]:

model_path = "../models/final_models_nifl"
fig_path = config["PATHS"]["figure_path"]
fnames = sorted([
    oj(model_path, fname) for fname in os.listdir(model_path) if "pkl" in fname
])
results_list = [pd.Series(pkl.load(open(fname, "rb"))) for fname in (fnames)]

results_all = pd.concat(results_list, axis=1).T.infer_objects()

results_all.experiment_name.unique()

exp_name = "NumCycles"
results = results_all[results_all.experiment_name == exp_name]
results = results.reset_index()

my_results = None

if results.experiment_name[0] == "NumCycles":

    my_results = (results[[
        "sequence_length",
        "best_val_loss",
        "rmse_state_val",
        "rmse_state_test",
        "dropout",
        "hidden_size",
        "hidden_size_lstm",
    ]].groupby([
        "sequence_length",
        "hidden_size",
        "hidden_size_lstm",
    ]).mean())

for seq_length in [100, 40]:
    exp_name = "NumCycles"
    results = results_all[results_all.experiment_name == exp_name]
    results = results.reset_index()

    results = results[results.dropout == 0.0]

    results = results[results.hidden_size_lstm == 32]
    results = results[results.hidden_size == 32]
    results = results[results.sequence_length == seq_length]
    results = results.sort_values("seed")
    results = results.reset_index()

    # In[9]:

    best_model_idx = results.rmse_state_val.argmin()

    start_cycle = results.start[best_model_idx]

    hidden_size = results.hidden_size[best_model_idx]

    # # load data

    # In[10]:

    if "data_dict" not in locals():  # just takes a lot of time
        data_path = config["DATASET"]["severson_path"]
        bat_dicts = severson_data.load_data_single(data_path)
    data_dict = {
        **bat_dicts[0],
        **bat_dicts[1],
    }
    # data_dict = {**bat_dicts[0], **bat_dicts[1], **bat_dicts[2],}

    x, y, c, var = severson_data.get_capacity_input(data_dict,
                                                    num_offset=0,
                                                    start_cycle=start_cycle,
                                                    stop_cycle=seq_length)

    x_scaled = severson_data.scale_x(x, y)

    x_preprocessed = severson_data.remove_outliers(x_scaled, y)

    x_smoothed = severson_data.smooth_x(x_preprocessed, y, num_points=20)

    # In[11]:

    train_idxs, val_idxs, test_idxs = severson_data.get_split(len(x), seed=42)

    qc_variance_scaler = StandardScaler().fit(var[train_idxs])
    var = qc_variance_scaler.transform(var)

    augmented_data = np.hstack([c, var])

    # In[12]:

    train_x, train_y, train_s = severson_data.assemble_dataset(
        x_preprocessed[train_idxs],
        y[train_idxs],
        augmented_data[train_idxs],
        seq_len=seq_length,
    )
    _, smoothed_y, _ = severson_data.assemble_dataset(
        x_smoothed[train_idxs],
        y[train_idxs],
        augmented_data[train_idxs],
        seq_len=seq_length,
    )

    min_val = 0.85
    max_val = 1

    capacity_output_scaler = MinMaxScaler((-1, 1), clip=False).fit(
        np.maximum(np.minimum(smoothed_y[:, 0:1], max_val), min_val))

    # In[13]:

    input_dim = train_x.shape[
        2]  # Number of input features (e.g. discharge capacity)
    num_augment = train_s.shape[
        1]  # three  values of charging schedule (avg and last) plus the variance

    my_models = [
        models.Uncertain_LSTM(
            train_x.shape[2],
            train_s.shape[1],
            num_hidden=results.iloc[i].hidden_size,
            num_hidden_lstm=results.iloc[i].hidden_size_lstm,
            seq_len=results.sequence_length[best_model_idx],
            n_layers=2,
            dropout=0.0,
        ).to(device) for i, _ in enumerate(results.file_name)
    ]
    for i, file_name in enumerate(results.file_name):
        my_models[i].load_state_dict(
            torch.load(oj(model_path, file_name + ".pt")))
        my_models[i] = my_models[i].to(device)

    # In[146]:

    # for use_ensemble, use_sample in [(True, True), (True, False), (False, True)]:
    for use_ensemble, use_sample in [
        (True, True),
    ]:  # (True, False), (False, True)]:
        my_quantile = 0.05

        if use_ensemble == True and use_sample == True:
            num_samples = 10
            num_models = 5
        elif use_ensemble == True:
            num_samples = 1
            num_models = 5
        elif use_sample == True:

            num_samples = 50
            num_models = 1

        # In[147]:

        np.random.seed(42)
        ensemble_idx = np.random.choice(5)

        # In[148]:

        test_seq_list = []
        test_seq_upper_quantile_list = []
        test_seq_lower_quantile_list = []
        test_life_pred_list = []
        test_seq_std_list = []
        used_idxs = test_idxs  # for actually new data, use test_idxs
        for model in tqdm(my_models[ensemble_idx * 5:ensemble_idx * 5 +
                                    num_models]):

            supp_val_data = np.hstack([
                c[used_idxs, :3],
                var[used_idxs],
                np.ones((len(used_idxs), 1)) * np.log(seq_length),
            ])

            test_seq = x_preprocessed[used_idxs][:, :seq_length, None].copy()
            extended_seq = np.swapaxes(
                np.reshape(
                    np.repeat(
                        np.swapaxes(test_seq, 0, -1)[:, :, :, None],
                        num_samples,
                        axis=-1,
                    ),
                    (1, seq_length, -1),
                ),
                0,
                -1,
            )

            extended_supp_data = np.swapaxes(
                np.reshape(
                    np.repeat(
                        np.swapaxes(supp_val_data, 0, -1)[:, :, None],
                        num_samples,
                        axis=-1,
                    ),
                    (supp_val_data.shape[1], -1),
                ),
                0,
                -1,
            )

            with torch.no_grad():
                while (
                        extended_seq.shape[1] < 3500
                ):  # ((np.all(extended_seq[:,-1] < 1e-3) == False ) *(extended_seq.shape[1] < 3500)):

                    supp_val_data_torch = (torch.from_numpy(
                        extended_supp_data).to(device).float())

                    test_seq_torch = (torch.from_numpy(
                        extended_seq[:, -seq_length:]).to(device).float())

                    model.reset_hidden_state()
                    (state_mean_mean,
                     state_var) = model(test_seq_torch, supp_val_data_torch)
                    if num_samples > 1:
                        state_mean_noisy = state_mean_mean + torch.normal(
                            0, (torch.sqrt(state_var)))
                    else:
                        state_mean_noisy = state_mean_mean

                    state_mean_transformed = torch.from_numpy(
                        capacity_output_scaler.inverse_transform(
                            state_mean_noisy.cpu().numpy())).to(device)

                    state_mean_transformed[:,
                                           0] = state_mean_transformed[:, 0] * (
                                               test_seq_torch[:, -1, 0])

                    extended_supp_data[:, -1] = np.log(
                        np.exp(extended_supp_data[:, -1]) + 1)
                    extended_seq = np.hstack([
                        extended_seq,
                        state_mean_transformed.cpu().numpy()[:, None]
                    ])
            used_steps = extended_seq.shape[1]
            reshaped = np.swapaxes(
                np.reshape(np.swapaxes(extended_seq, 0, 1),
                           (1, used_steps, -1, num_samples)),
                0,
                -2,
            )
            test_seq_list.append(reshaped[:, :, 0, :])
            test_seq_std_list.append(reshaped.std(axis=-1))
            test_seq_lower_quantile_list.append(
                np.quantile(reshaped, my_quantile / 2, axis=-1))
            test_seq_upper_quantile_list.append(
                np.quantile(reshaped, 1 - my_quantile / 2, axis=-1))

        max_used_steps = max([x.shape[1] for x in test_seq_list])

        # In[151]:

        my_quantile = 0.05

        all_outputs = np.concatenate(test_seq_list,
                                     axis=-1)[:, :, :num_models * num_samples]
        mean = all_outputs.mean(axis=-1) * (1.1 - 0.8 * 1.1) + 0.8 * 1.1
        all_lower_quantile = (
            np.quantile(all_outputs, my_quantile / 2, axis=-1) *
            (1.1 - 0.8 * 1.1) + 0.8 * 1.1)
        all_upper_quantile = (
            np.quantile(all_outputs, 1 - my_quantile / 2, axis=-1) *
            (1.1 - 0.8 * 1.1) + 0.8 * 1.1)
        mean = np.quantile(all_outputs, 0.5,
                           axis=-1) * (1.1 - 0.8 * 1.1) + 0.8 * 1.1

        pickle_results_dict = {}
        pickle_results_dict["true_sequence "] = (x_preprocessed[used_idxs] *
                                                 (1.1 - 0.8 * 1.1) + 0.8 * 1.1)
        pickle_results_dict["fifty_percentile"] = mean
        pickle_results_dict["upper_percentile"] = all_upper_quantile
        pickle_results_dict["lower_percentile"] = all_lower_quantile

        fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(20, 10))
        linewidth = 4

        for i, ax in enumerate(fig.get_axes()):
            if i >= len(used_idxs):
                break

            ax.plot(
                x_preprocessed[used_idxs[i]] * (1.1 - 0.8 * 1.1) + 0.8 * 1.1,
                c=colours[1],
                label="Data",
                linewidth=linewidth,
            )

            ax.fill_between(
                np.arange(max_used_steps),
                all_upper_quantile[i],
                all_lower_quantile[i],
                facecolor=colours[0],
                alpha=0.2,
            )
            ax.plot(mean[i],
                    c=colours[0],
                    label="Predicted",
                    linewidth=linewidth)
            ax.set_yticks([0.8 * 1.1, 0.9 * 1.1, 1.1])

            ax.set_ylim(0.8 * 1.1 + 0.002, 1.1)

            ax.set_xlim(
                0, 2500
            )  # np.maximum(y[used_idxs[i]], (all_outputs[model_idx, i,:] < 1e-3).argmax())+50)
            ax.axvline(x=seq_length,
                       linestyle="--",
                       c="k",
                       label="Used for prediction")
        axes[-2, -1].legend(loc=1)
        plt.tight_layout()
        if use_ensemble == True and use_sample == True:
            plt.savefig(
                oj(
                    fig_path,
                    "uncertainEnsembleLSTM_{0!s}.pdf".format(seq_length, ),
                ))
            plt.savefig(
                oj(
                    fig_path,
                    "uncertainEnsembleLSTM_{0!s}.png".format(seq_length, ),
                ))
            with open(
                    oj(
                        fig_path,
                        "uncertainEnsembleLSTM_{0!s}.pickle".format(
                            seq_length, ),
                    ),
                    "wb",
            ) as handle:
                pkl.dump(
                    pickle_results_dict,
                    handle,
                )
        elif use_ensemble == True:

            plt.savefig(
                oj(
                    fig_path,
                    "ensembleLSTM_{0!s}.pdf".format(seq_length, ),
                ))
            plt.savefig(
                oj(
                    fig_path,
                    "ensembleLSTM_{0!s}.png".format(seq_length, ),
                ))
            with open(
                    oj(
                        fig_path,
                        "ensembleLSTM_{0!s}.pickle".format(seq_length, ),
                    ),
                    "wb",
            ) as handle:
                pkl.dump(
                    pickle_results_dict,
                    handle,
                )
        elif use_sample == True:

            plt.savefig(
                oj(
                    fig_path,
                    "uncertainLSTM_{0!s}.pdf".format(seq_length, ),
                ))
            plt.savefig(
                oj(
                    fig_path,
                    "uncertainLSTM_{0!s}.png".format(seq_length, ),
                ))

            with open(
                    oj(
                        fig_path,
                        "uncertainLSTM_{0!s}.pickle".format(seq_length, ),
                    ),
                    "wb",
            ) as handle:
                pkl.dump(
                    pickle_results_dict,
                    handle,
                )

        aged_data_dict = {
            **bat_dicts[2],
        }
        aged_x, aged_y, aged_c, aged_var = severson_data.get_capacity_input(
            aged_data_dict,
            num_offset=0,
            start_cycle=start_cycle,
            stop_cycle=seq_length)

        aged_test_idxs = np.arange(len(aged_x))

        aged_var = qc_variance_scaler.transform(aged_var)
        aged_augmented_data = np.hstack([aged_c, aged_var])

        aged_x = severson_data.remove_outliers(
            severson_data.scale_x(aged_x, aged_y), aged_y)

        old_aged_x = aged_x.copy()

        aged_idxs = np.arange(len(aged_x))
        aged_test_x, aged_test_y, aged_test_s = severson_data.assemble_dataset(
            aged_x, aged_y, aged_augmented_data, seq_len=seq_length)
        aged_test_y[:,
                    0:1] = capacity_output_scaler.transform(aged_test_y[:,
                                                                        0:1])

        aged_dataset = TensorDataset(*[
            torch.Tensor(input)
            for input in [aged_test_x, aged_test_y, aged_test_s]
        ])  # create your datset
        aged_test_loader = DataLoader(aged_dataset,
                                      batch_size=256,
                                      shuffle=True)
        input_dim = train_x.shape[
            2]  # Number of input features (e.g. discharge capacity)
        num_augment = train_s.shape[
            1]  # three  values of charging schedule (avg and last) plus the variance

        test_seq_list = []
        test_seq_upper_quantile_list = []
        test_seq_lower_quantile_list = []
        test_life_pred_list = []
        test_seq_std_list = []
        used_idxs = aged_idxs
        for model in tqdm(my_models[ensemble_idx * 5:ensemble_idx * 5 +
                                    num_models]):

            supp_val_data = np.hstack([
                aged_c[used_idxs, :3],
                aged_var[used_idxs],
                np.ones((len(used_idxs), 1)) * np.log(seq_length),
            ])

            test_seq = old_aged_x[used_idxs][:, :seq_length, None].copy()
            extended_seq = np.swapaxes(
                np.reshape(
                    np.repeat(
                        np.swapaxes(test_seq, 0, -1)[:, :, :, None],
                        num_samples,
                        axis=-1,
                    ),
                    (1, seq_length, -1),
                ),
                0,
                -1,
            )

            extended_supp_data = np.swapaxes(
                np.reshape(
                    np.repeat(
                        np.swapaxes(supp_val_data, 0, -1)[:, :, None],
                        num_samples,
                        axis=-1,
                    ),
                    (supp_val_data.shape[1], -1),
                ),
                0,
                -1,
            )

            with torch.no_grad():
                while (
                        extended_seq.shape[1] < 3500
                ):  # ((np.all(extended_seq[:,-1] < 1e-3) == False ) *(extended_seq.shape[1] < 3500)):

                    supp_val_data_torch = (torch.from_numpy(
                        extended_supp_data).to(device).float())

                    test_seq_torch = (torch.from_numpy(
                        extended_seq[:, -seq_length:]).to(device).float())

                    model.reset_hidden_state()
                    (state_mean_mean,
                     state_var) = model(test_seq_torch, supp_val_data_torch)
                    if num_samples > 1:
                        state_mean_noisy = state_mean_mean + torch.normal(
                            0, (torch.sqrt(state_var)))
                    else:
                        state_mean_noisy = state_mean_mean

                    state_mean_transformed = torch.from_numpy(
                        capacity_output_scaler.inverse_transform(
                            state_mean_noisy.cpu().numpy())).to(device)

                    state_mean_transformed[:,
                                           0] = state_mean_transformed[:, 0] * (
                                               test_seq_torch[:, -1, 0])

                    extended_supp_data[:, -1] = np.log(
                        np.exp(extended_supp_data[:, -1]) + 1)
                    extended_seq = np.hstack([
                        extended_seq,
                        state_mean_transformed.cpu().numpy()[:, None]
                    ])
            used_steps = extended_seq.shape[1]
            reshaped = np.swapaxes(
                np.reshape(np.swapaxes(extended_seq, 0, 1),
                           (1, used_steps, -1, num_samples)),
                0,
                -2,
            )
            test_seq_list.append(reshaped[:, :, 0, :])
            test_seq_std_list.append(reshaped.std(axis=-1))
            test_seq_lower_quantile_list.append(
                np.quantile(reshaped, my_quantile / 2, axis=-1))
            test_seq_upper_quantile_list.append(
                np.quantile(reshaped, 1 - my_quantile / 2, axis=-1))

        max_used_steps = max([x.shape[1] for x in test_seq_list])

        all_outputs = np.concatenate(test_seq_list,
                                     axis=-1)[:, :, :num_models * num_samples]
        # mean = all_outputs.mean(axis=-1)* (1.1-.8*1.1) + .8*1.1
        all_lower_quantile = (
            np.quantile(all_outputs, my_quantile / 2, axis=-1) *
            (1.1 - 0.8 * 1.1) + 0.8 * 1.1)
        all_upper_quantile = (
            np.quantile(all_outputs, 1 - my_quantile / 2, axis=-1) *
            (1.1 - 0.8 * 1.1) + 0.8 * 1.1)
        mean = np.quantile(all_outputs, 0.5,
                           axis=-1) * (1.1 - 0.8 * 1.1) + 0.8 * 1.1

        pickle_results_dict = {}
        pickle_results_dict["true_sequence "] = (old_aged_x[used_idxs] *
                                                 (1.1 - 0.8 * 1.1) + 0.8 * 1.1)
        pickle_results_dict["fifty_percentile"] = mean
        pickle_results_dict["upper_percentile"] = all_upper_quantile
        pickle_results_dict["lower_percentile"] = all_lower_quantile

        fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(20, 10))
        linewidth = 4

        for i, ax in enumerate(fig.get_axes()):
            if i >= len(used_idxs):
                break

            ax.plot(
                old_aged_x[used_idxs[i]] * (1.1 - 0.8 * 1.1) + 0.8 * 1.1,
                c=colours[1],
                label="Data",
                linewidth=linewidth,
            )

            ax.fill_between(
                np.arange(max_used_steps),
                all_upper_quantile[i],
                all_lower_quantile[i],
                facecolor=colours[0],
                alpha=0.2,
            )
            ax.plot(mean[i],
                    c=colours[0],
                    label="Predicted",
                    linewidth=linewidth)
            ax.set_yticks([0.8 * 1.1, 0.9 * 1.1, 1.1])

            ax.set_ylim(0.8 * 1.1 + 0.002, 1.1)

            ax.set_xlim(
                0, 2500
            )  # np.maximum(y[used_idxs[i]], (all_outputs[model_idx, i,:] < 1e-3).argmax())+50)
            ax.axvline(x=seq_length,
                       linestyle="--",
                       c="k",
                       label="Used for prediction")
        axes[-2, -1].legend(loc=1)
        plt.tight_layout()
        if use_ensemble == True and use_sample == True:
            plt.savefig(
                oj(
                    fig_path,
                    "agedUncertainEnsembleLSTM_{0!s}.pdf".format(seq_length, ),
                ))
            plt.savefig(
                oj(
                    fig_path,
                    "agedUncertainEnsembleLSTM_{0!s}.png".format(seq_length, ),
                ))

            with open(
                    oj(
                        fig_path,
                        "agedUncertainEnsembleLSTM_{0!s}.pickle".format(
                            seq_length, ),
                    ),
                    "wb",
            ) as handle:
                pkl.dump(
                    pickle_results_dict,
                    handle,
                )

        elif use_ensemble == True:

            plt.savefig(
                oj(
                    fig_path,
                    "agedEnsembleLSTM_{0!s}.pdf".format(seq_length, ),
                ))
            plt.savefig(
                oj(
                    fig_path,
                    "agedEnsembleLSTM_{0!s}.png".format(seq_length, ),
                ))
            with open(
                    oj(
                        fig_path,
                        "agedEnsembleLSTM_{0!s}.pickle".format(seq_length, ),
                    ),
                    "wb",
            ) as handle:
                pkl.dump(
                    pickle_results_dict,
                    handle,
                )

        elif use_sample == True:

            plt.savefig(
                oj(
                    fig_path,
                    "agedUncertainLSTM_{0!s}.pdf".format(seq_length, ),
                ))
            plt.savefig(
                oj(
                    fig_path,
                    "agedUncertainLSTM_{0!s}.png".format(seq_length, ),
                ))
            with open(
                    oj(
                        fig_path,
                        "agedUncertainLSTM_{0!s}.pickle".format(seq_length, ),
                    ),
                    "wb",
            ) as handle:
                pkl.dump(
                    pickle_results_dict,
                    handle,
                )
