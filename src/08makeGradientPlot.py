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

import my_eval

sys.path.insert(0, "../src_lstm")
import pickle as pkl

import seaborn as sns
from kneed import DataGenerator, KneeLocator

import models
import my_eval
import severson_data

pd.set_option("display.float_format", lambda x: "%.2f" % x)
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


# In[3]:


colours = ["#208F90", "#8F2317", "#17608F", "#8F5F17", "#f2f3f4", "#E56399", "#DE6E4B"]
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
fnames = sorted(
    [oj(model_path, fname) for fname in os.listdir(model_path) if "pkl" in fname]
)
results_list = [pd.Series(pkl.load(open(fname, "rb"))) for fname in (fnames)]


# In[6]:


results_all = pd.concat(results_list, axis=1).T.infer_objects()

results = results_all[results_all.experiment_name == "NumCycles"]
results = results.reset_index()


# In[7]:


my_sue = "NumCycles"
results = results_all[results_all.experiment_name == my_sue]
results = results.reset_index()


# In[8]:


results = results[results.dropout == 0.0]

results = results[results.hidden_size_lstm == 32]
results = results[results.hidden_size == 32]
results = results[results.sequence_length == 100]
results = results.sort_values("seed")
results = results.reset_index()


# In[9]:


best_model_idx = results.rmse_state_val.argmin()
seq_length = int(results.sequence_length[best_model_idx])
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

x, y, c, var = severson_data.get_capacity_input(
    data_dict, num_offset=0, start_cycle=start_cycle, stop_cycle=100
)

x = severson_data.scale_x(x, y)

x = severson_data.remove_outliers(x, y)
old_x = x.copy()

smoothed_x = severson_data.smooth_x(x, y, num_points=20)


# In[11]:


train_idxs, val_idxs, test_idxs = severson_data.get_split(len(x), seed=42)

qc_variance_scaler = StandardScaler().fit(var[train_idxs])
var = qc_variance_scaler.transform(var)

augmented_data = np.hstack([c, var])


# In[12]:


train_x, train_y, train_s = severson_data.assemble_dataset(
    smoothed_x[train_idxs],
    y[train_idxs],
    augmented_data[train_idxs],
    seq_len=seq_length,
)

min_val = 0.85
max_val = 1

capacity_output_scaler = MinMaxScaler((-1, 1), clip=False).fit(
    np.maximum(np.minimum(train_y[:, 0:1], max_val), min_val)
)
input_dim = train_x.shape[2]  # Number of input features (e.g. discharge capacity)
num_augment = train_s.shape[
    1
]  # three  values of charging schedule (avg and last) plus the variance


# # Sensitivity

# In[13]:


my_models = [
    models.Uncertain_LSTM_Module(
        train_x.shape[2],
        train_s.shape[1],
        num_hidden=results.iloc[i].hidden_size,
        num_hidden_lstm=results.iloc[i].hidden_size_lstm,
        seq_len=results.sequence_length[best_model_idx],
        n_layers=2,
        dropout=0.0,
    ).to(device)
    for i, _ in enumerate(results.file_name)
]
for i, file_name in enumerate(results.file_name):
    my_models[i].load_state_dict(torch.load(oj(model_path, file_name + ".pt")))
    my_models[i] = my_models[i].to(device)
# iiidx = 14

cur_model = my_models[0]


importance_list = []
test_seq_list = []
test_life_pred_list = []
used_idxs = test_idxs  # for actually new data, use test_idxs


supp_val_data = np.hstack(
    [
        c[used_idxs, :3],
        var[used_idxs],
        np.ones((len(used_idxs), 1)) * np.log(seq_length),
    ]
)

test_seq = old_x[used_idxs][:, :seq_length, None].copy()


while (np.all(test_seq[:, -1] < 10e-3) == False) * (test_seq.shape[1] < 3500):

    supp_val_data_torch = torch.from_numpy(supp_val_data).to(device).float()
    supp_val_data_torch.requires_grad = True
    test_seq_torch = torch.from_numpy(test_seq[:, -seq_length:]).to(device).float()
    test_seq_torch.requires_grad = True
    cur_model.reset_hidden_state()
    (pred_state, pred_var) = cur_model(test_seq_torch, supp_val_data_torch)
    augmented_grad = (
        torch.autograd.grad(pred_state.sum(), supp_val_data_torch, retain_graph=True)[0]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )
    sequence_grad = (
        torch.autograd.grad(pred_state.sum(), test_seq_torch, retain_graph=False)[0]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )
    all_grads = np.concatenate(
        [
            np.abs(sequence_grad[:, -1:, 0]),
            np.abs(augmented_grad),
        ],
        axis=1,
    )
    
    # sequence_grad = (
    #     torch.autograd.grad(pred_state.sum(), cur_model.intermediate_output, retain_graph=False)[0]
    #     .detach()
    #     .cpu()
    #     .numpy()
    #     .copy()
    # )


    # all_grads = np.concatenate(
    #     [
    #         np.abs(sequence_grad[:, :32]).mean(axis=1)[:,None],
    #         np.abs(sequence_grad[:, -7:]),
    #     ],
    #     axis=1,
    # )

    importance_list.append(all_grads)
    pred_state = torch.from_numpy(
        capacity_output_scaler.inverse_transform(pred_state.detach().cpu().numpy())
    ).to(device)

    pred_state[:, 0] = pred_state[:, 0] * test_seq_torch[:, -1, 0]
    supp_val_data[:, -1] = np.log(np.exp(supp_val_data[:, -1]) + 1)
    test_seq = np.hstack([test_seq, pred_state.detach().cpu().numpy()[:, None]])

test_seq_list.append(test_seq)


all_outputs = np.asarray(test_seq_list)[:, :, :, 0]
predicted_y = (all_outputs[0] < 10e-3).argmax(axis=1)


num_cycles_before_eol = 300
num_ticks = int(num_cycles_before_eol / 50) + 1


# In[16]:


all_grads_dying = np.zeros((len(used_idxs), 8, num_cycles_before_eol))
for i, bat_idx in enumerate(used_idxs):
    for j in range(num_cycles_before_eol):

        if predicted_y[i] - 100 - num_cycles_before_eol + j < 0:
            print("GG")
            break
        all_grads_dying[i, :, j] = importance_list[
            predicted_y[i] - 100 - num_cycles_before_eol + j
        ][i]


arr_of_curves = np.zeros((len(predicted_y), num_cycles_before_eol))
for idx in range(len(predicted_y)):
    arr_of_curves[idx] = (
        all_outputs[0, idx, predicted_y[idx] - num_cycles_before_eol : predicted_y[idx]]
        * (1.1 - 0.8 * 1.1)
        + 0.8 * 1.1
    )


linewidth = 3
fontsize = 15

show_idxs = np.where(predicted_y < 6000)[0]
fig, axes = plt.subplots(
    nrows=4,
    figsize=(8, 5),
    gridspec_kw={"height_ratios": [1, 1.0, 1.0, 1]},
    sharex=True,
)
axes[0].plot(arr_of_curves[0], c=".75", label="Capacity curves")
for i in range(len(predicted_y)):

    axes[0].plot(arr_of_curves[i], c=str(0.75))


# for i in show_idxs:
# axes[1].plot((all_grads_dying[i,0] / all_grads_dying[i].sum(axis=0)), linewidth = linewidth, label = 'Previous capacities')

mean_plot = (
    all_grads_dying[show_idxs, 0] / all_grads_dying[show_idxs].sum(axis=1)
).mean(axis=0)
std_plot = (all_grads_dying[show_idxs, 0] / all_grads_dying[show_idxs].sum(axis=1)).std(
    axis=0
)

# stuff to make plot broken up
for i in range(1, 3):
    axes[i].plot(
        (
            all_grads_dying[show_idxs, 4:-1].sum(axis=1)
            / all_grads_dying[show_idxs].sum(axis=1)
        ).mean(axis=0),
        linewidth=linewidth,
        label="In-cycle info",
    )
    axes[i].plot(
        (all_grads_dying[show_idxs, 0] / all_grads_dying[show_idxs].sum(axis=1)).mean(
            axis=0
        ),
        linewidth=linewidth,
        label="Previous capacities",
    )
    axes[i].plot(
        (
            all_grads_dying[show_idxs, 1:4].sum(axis=1)
            / all_grads_dying[show_idxs].sum(axis=1)
        ).mean(axis=0),
        linewidth=linewidth,
        label="C-Rate",
    )

    axes[i].plot(
        (all_grads_dying[show_idxs, -1] / all_grads_dying[show_idxs].sum(axis=1)).mean(
            axis=0
        ),
        linewidth=linewidth,
        label="Cycle counter",
        linestyle="dashed",
    )


axes[1].set_ylim(0.65, 0.9)
axes[2].set_ylim(0.0, 0.2)

axes[1].spines["bottom"].set_visible(False)
axes[2].spines["top"].set_visible(False)
axes[1].xaxis.tick_top()
axes[1].tick_params(labeltop=False)  # don't put tick labels at the top
axes[2].xaxis.tick_bottom()

d = 0.015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=axes[1].transAxes, color="k", clip_on=False)
axes[1].plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
axes[1].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=axes[2].transAxes)  # switch to the bottom axes
axes[2].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
axes[2].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


axes[1].yaxis.set_label_coords(-0.1, 0.3)

# end stuff


axes[1].legend(bbox_to_anchor=(1.1, 1), loc="upper left")
axes[0].legend(bbox_to_anchor=(1.0, 1.03), loc="upper left")
axes[0].set_ylabel("Capacity\n", fontsize=fontsize)
axes[2].set_ylabel("Feature importance\n", fontsize=fontsize)
# axes[3].set_ylabel("Feature\n importance",fontsize = fontsize)
axes[3].set_xlabel("Cycles to EOL", fontsize=fontsize)
axes[1].set_xticks([50 * x for x in range(num_ticks)])
axes[0].set_xticks([50 * x for x in range(num_ticks)])
axes[0].set_xticklabels([], fontsize=fontsize * 0.8)
axes[1].set_xticklabels([], fontsize=fontsize * 0.8)
axes[1].set_xlim(0, num_cycles_before_eol)
axes[0].set_xlim(0, num_cycles_before_eol)
axes[3].set_xlim(0, num_cycles_before_eol)
axes[3].set_ylim(0, 0.14)
axes[3].set_xticklabels(
    [-num_cycles_before_eol + 50 * x for x in range(num_ticks)], fontsize=fontsize * 0.8
)
show_idxs = np.where(predicted_y < 700)[0]
axes[3].plot(
    (all_grads_dying[show_idxs, 5] / all_grads_dying[show_idxs].sum(axis=1)).mean(
        axis=0
    ),
    linewidth=0.7 * linewidth,
    label="Coloumbic Eff. (EOL < $\mathregular{EOL_{Avg}}$)",
)
show_idxs = np.where(y[used_idxs] > 700)[0]
axes[3].plot(
    (all_grads_dying[show_idxs, 5] / all_grads_dying[show_idxs].sum(axis=1)).mean(
        axis=0
    ),
    linewidth=0.7 * linewidth,
    label="Coloumbic Eff. (EOL > $\mathregular{EOL_{Avg}}$)",
)
axes[3].legend(bbox_to_anchor=(1.0, 1.03), loc="upper left")

plt.tight_layout()

plt.savefig(
    oj(fig_path, "Sensitivity_analysis.pdf"),
)
plt.savefig(
    oj(fig_path, "Sensitivity_analysis.png"),
)
gradient_dict = {}
gradient_dict["all_grads_dying"] = all_grads_dying
gradient_dict["arr_of_curves"] = arr_of_curves

gradient_dict["predicted_y"] = predicted_y
with open(oj(result_path, "gradients.pickle"), "wb") as handle:
    pkl.dump(
        gradient_dict,
        handle,
    )
