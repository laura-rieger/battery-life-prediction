#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[7]:


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


# In[8]:


config = configparser.ConfigParser()
config.read("../config.ini")
result_path = config["PATHS"]["result_path"]


# In[9]:


model_path = "../models/final_models_nifl"
fig_path = config["PATHS"]["figure_path"]
fnames = sorted(
    [oj(model_path, fname) for fname in os.listdir(model_path) if "pkl" in fname]
)
results_list = [pd.Series(pkl.load(open(fname, "rb"))) for fname in (fnames)]


# In[10]:


results_all = pd.concat(results_list, axis=1).T.infer_objects()

results_all.experiment_name.unique()


# In[11]:


my_sue = "NumCycles"
results = results_all[results_all.experiment_name == my_sue]
results = results[results.dropout == 0.0]

results = results[results.hidden_size_lstm == 32]
results = results[results.hidden_size == 32]
results = results[results.sequence_length == 100]
results = results.sort_values("seed")
results = results.reset_index()


# In[12]:


best_model_idx = results.rmse_state_val.argmin()
seq_length = int(results.sequence_length[best_model_idx])
start_cycle = results.start[best_model_idx]


hidden_size = results.hidden_size[best_model_idx]


# # load data

# In[13]:


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

x_scaled = severson_data.scale_x(x, y)

x_preprocessed = severson_data.remove_outliers(x_scaled, y)
# x_splined = severson_data.spline_x(x_preprocessed, y)
x_smoothed = severson_data.smooth_x(x_preprocessed, y, num_points=20)


# In[14]:


train_idxs, val_idxs, test_idxs = severson_data.get_split(len(x), seed=42)

qc_variance_scaler = StandardScaler().fit(var[train_idxs])
var = qc_variance_scaler.transform(var)

augmented_data = np.hstack([c, var])


# In[15]:


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
    np.maximum(np.minimum(smoothed_y[:, 0:1], max_val), min_val)
)


# In[16]:


input_dim = train_x.shape[2]  # Number of input features (e.g. discharge capacity)
num_augment = train_s.shape[
    1
]  # three  values of charging schedule (avg and last) plus the variance


# In[17]:


aged_data_dict = {
    **bat_dicts[2],
}
aged_x, aged_y, aged_c, aged_var = severson_data.get_capacity_input(
    aged_data_dict, num_offset=0, start_cycle=start_cycle, stop_cycle=seq_length
)

aged_test_idxs = np.arange(len(aged_x))

aged_var = qc_variance_scaler.transform(aged_var)
aged_augmented_data = np.hstack([aged_c, aged_var])


aged_x = severson_data.preprocess_x(aged_x, aged_y)

old_aged_x = aged_x.copy()


# # Sensitivity

# In[57]:


my_models = [
    models.Uncertain_LSTM(
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
cur_model = my_models[0]  # results.best_val_loss.argmin()]


# In[58]:


num_used_eol = 250


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

test_seq = x_preprocessed[used_idxs][:, :seq_length, None].copy()


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
            np.abs(sequence_grad[:, :, 0][:, -1:]),
            np.abs(augmented_grad),
        ],
        axis=1,
    )
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


# In[60]:


all_grads_dying = np.zeros((len(used_idxs), 8, num_used_eol))
all_grads_living = np.zeros((len(used_idxs), 8, num_used_eol))
for i, bat_idx in enumerate(used_idxs):
    for j in range(num_used_eol):
        #         print(i,j)
        all_grads_living[i, :, j] = importance_list[j][i]
        importance_list[predicted_y[i] - num_used_eol + j - seq_length]
        all_grads_dying[i, :, j] = importance_list[
            predicted_y[i] - num_used_eol + j - seq_length
        ][i]


# # old data eval

# In[61]:


aged_idxs = np.arange(len(aged_x))
aged_test_x, aged_test_y, aged_test_s = severson_data.assemble_dataset(
    aged_x, aged_y, aged_augmented_data, seq_len=seq_length
)
aged_test_y[:, 0:1] = capacity_output_scaler.transform(aged_test_y[:, 0:1])


# In[62]:


aged_dataset = TensorDataset(
    *[torch.Tensor(input) for input in [aged_test_x, aged_test_y, aged_test_s]]
)  # create your datset
aged_test_loader = DataLoader(aged_dataset, batch_size=256, shuffle=True)
input_dim = train_x.shape[2]  # Number of input features (e.g. discharge capacity)
num_augment = train_s.shape[
    1
]  # three  values of charging schedule (avg and last) plus the variance


# In[63]:


importance_list = []
test_seq_list = []
test_life_pred_list = []
used_idxs = aged_idxs  # for actually new data, use test_idxs


supp_val_data = np.hstack(
    [
        aged_c[used_idxs, :3],
        aged_var[used_idxs],
        np.ones((len(used_idxs), 1)) * np.log(seq_length),
    ]
)

test_seq = aged_x[used_idxs][:, :seq_length, None].copy()


while (np.all(test_seq[:, -1] < 1e-3) == False) * (test_seq.shape[1] < 3500):

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
            np.abs(sequence_grad[:, :, 0][:, -1:]),
            np.abs(augmented_grad),
        ],
        axis=1,
    )
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


# In[64]:


grad_steps = num_used_eol
aged_grads_dying = np.zeros((len(used_idxs), 8, grad_steps))
aged_grads_living = np.zeros((len(used_idxs), 8, grad_steps))

for i, bat_idx in enumerate(used_idxs):
    for j in range(grad_steps):
        aged_grads_living[i, :, j] = importance_list[j][i]
        aged_grads_dying[i, :, j] = importance_list[
            predicted_y[bat_idx] - grad_steps - seq_length + j
        ][i]


# In[65]:


interval = 95
upper_limit = 100 - (100 - interval) / 2
lower_limit = 0 + (100 - interval) / 2




filter_battery_eol = y.mean()



# excluding all normal batteries with less life time since this would water up results


fig, ax = plt.subplots()

# excluding all normal batteries with less life time since this would water up results

linewidth = 4.5
fontsize = 20
show_idxs = np.where(y[test_idxs] > filter_battery_eol)[0]
mean_plot = (
    all_grads_dying[show_idxs, 4:].sum(axis=1)
    / all_grads_dying[show_idxs].sum(axis=1)
).mean(axis=0)
std_plot = (
    all_grads_dying[show_idxs, 4:].sum(axis=1)
    / all_grads_dying[show_idxs].sum(axis=1)
).std(axis=0)


# plt.fill_between(np.arange(400), max_plot, min_plot , facecolor=colours[0], alpha=0.2)
ax.fill_between(
    np.arange(num_used_eol),
    mean_plot + std_plot,
    mean_plot - std_plot,
    facecolor=colours[0],
    alpha=0.2,
)

ax.plot(mean_plot,
    c=colours[0],
    linewidth=linewidth,
    label="Non aged",
)


mean_plot = (
    aged_grads_dying[:, 4:].sum(axis=1) / aged_grads_dying[:].sum(axis=1)
).mean(axis=0)
std_plot = (
    aged_grads_dying[:, 4:].sum(axis=1) / aged_grads_dying[:].sum(axis=1)
).std(axis=0)


ax.plot(
    mean_plot,
    c=colours[1],
    linewidth=linewidth,
    label="Aged",
)
# plt.fill_between(np.arange(400),  max_plot,  min_plot, facecolor=colours[1], alpha=0.2)
ax.fill_between(
    np.arange(num_used_eol),
    mean_plot + std_plot,
    mean_plot - std_plot,
    facecolor=colours[1],
    alpha=0.2,
)
plt.xlim(0, num_used_eol)
plt.xticks([50 * x for x in range(int(num_used_eol/50+1))], [-num_used_eol + 50 * x for x in range(int(num_used_eol/50+1))])

plt.legend()
# plt.title("Relative importance of in-cycle  info \n in the last 300 cycles before death for slow and fast dying")

labels = [item for item in ax.get_yticks()]
labels=[str(int(100*float(x))) + '%'  for x in labels]
ax.set_yticklabels(labels)

ax.set_ylabel("Importance  \n of covariates", fontsize=fontsize)
ax.set_xlabel("Cycles (from 250 before EOL)", fontsize=fontsize)
# plt.ylim(0, 0.3)
plt.tight_layout()
plt.savefig(oj(fig_path, "Sensitivity_analysis_aged_vs_nonaged_eol.pdf"))
plt.savefig(oj(fig_path, "Sensitivity_analysis_aged_vs_nonaged_eol.png"), dpi = 200)

gradient_dict = {}
gradient_dict["aged_grads_dying"] = aged_grads_dying
gradient_dict["all_grads_dying"] = all_grads_dying
gradient_dict["y_test"] = y[test_idxs]


with open(oj(result_path, "aged_gradients.pickle"), "wb") as handle:
    pkl.dump(
        gradient_dict,
        handle,
    )


# In[74]:


# random_grads_dying = np.concatenate([aged_grads_dying, all_grads_dying[show_idxs]])
# shuffle_idx = np.arange(len(random_grads_dying))
# np.random.seed(0)
# np.random.shuffle(shuffle_idx)
# num_split = int(len(random_grads_dying) / 2)
# random_grads_one = random_grads_dying[shuffle_idx[:num_split]]
# random_grads_two = random_grads_dying[shuffle_idx[num_split:]]


# # print(random_grads_dying)
# linewidth = 4.5
# fontsize = 20

# mean_plot = (
#     random_grads_one[:, 4:-1].sum(axis=1) / random_grads_one[:].sum(axis=1)
# ).mean(axis=0)
# std_plot = (
#     random_grads_one[:, 4:-1].sum(axis=1) / random_grads_one[:].sum(axis=1)
# ).std(axis=0)
# min_plot = np.percentile(
#     (random_grads_one[:, 4:-1].sum(axis=1) / random_grads_one[:].sum(axis=1)),
#     lower_limit,
#     axis=0,
# )
# max_plot = np.percentile(
#     (random_grads_one[:, 4:-1].sum(axis=1) / random_grads_one[:].sum(axis=1)),
#     upper_limit,
#     axis=0,
# )


# # plt.fill_between(np.arange(400), max_plot, min_plot , facecolor=colours[0], alpha=0.2)
# plt.fill_between(
#     np.arange(400),
#     mean_plot + std_plot,
#     mean_plot - std_plot,
#     facecolor=colours[0],
#     alpha=0.2,
# )

# plt.plot(
#     (random_grads_one[:, 4:-1].sum(axis=1) / random_grads_one[:].sum(axis=1)).mean(
#         axis=0
#     ),
#     c=colours[0],
#     linewidth=linewidth,
#     label="Random 1",
# )


# mean_plot = (
#     random_grads_two[:, 4:-1].sum(axis=1) / random_grads_two[:].sum(axis=1)
# ).mean(axis=0)
# std_plot = (
#     random_grads_two[:, 4:-1].sum(axis=1) / random_grads_two[:].sum(axis=1)
# ).std(axis=0)
# min_plot = np.percentile(
#     (random_grads_two[:, 4:-1].sum(axis=1) / random_grads_two[:].sum(axis=1)),
#     lower_limit,
#     axis=0,
# )
# max_plot = np.percentile(
#     (random_grads_two[:, 4:-1].sum(axis=1) / random_grads_two[:].sum(axis=1)),
#     upper_limit,
#     axis=0,
# )

# plt.plot(
#     (random_grads_two[:, 4:-1].sum(axis=1) / random_grads_two[:].sum(axis=1)).mean(
#         axis=0
#     ),
#     c=colours[1],
#     linewidth=linewidth,
#     label="Random 2",
# )
# # plt.fill_between(np.arange(400),  max_plot,  min_plot, facecolor=colours[1], alpha=0.2)
# plt.fill_between(
#     np.arange(400),
#     mean_plot + std_plot,
#     mean_plot - std_plot,
#     facecolor=colours[1],
#     alpha=0.2,
# )

# plt.xticks([50 * x for x in range(9)], [-400 + 50 * x for x in range(9)])

# plt.legend()
# # plt.title("Relative importance of in-cycle  info \n in the last 300 cycles before death for slow and fast dying")

# plt.ylabel("Importance  \n of in-cycle info", fontsize=fontsize)
# plt.xlabel("Cycles (from 400 before EOL)", fontsize=fontsize)
# plt.ylim(0, 0.3)
# plt.tight_layout()
# plt.savefig(oj(fig_path, "Sensitivity_analysis_random_vs_random.pdf"))

# plt.savefig(oj(fig_path, "Sensitivity_analysis_random_vs_random.png"))
