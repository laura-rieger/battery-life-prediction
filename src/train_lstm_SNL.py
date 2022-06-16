# in theory this could be in the same script as

import os
import sys

sys.path.insert(0, "src_lstm")
import argparse
import configparser
import pickle as pkl
import sys
import warnings
from argparse import ArgumentParser
from copy import deepcopy
from os import environ
from os.path import join as oj

import numpy as np
import torch
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import models
import my_eval
import severson_data
import snl_data
from loss_functions import nll_loss, mse_loss

warnings.filterwarnings("ignore", category=RuntimeWarning)
environ["CUBLAS_WORKSPACE_CONFIG"] = ""
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html LSTM have issues
# with non-deterministic behavior

cuda = torch.cuda.is_available()

device = torch.device("cuda")


def get_args():
    parser = ArgumentParser(description="Battery cycle prediction with LSTM")

    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--experiment_name", type=str, default="")

    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_size_lstm", type=int, default=-1)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--use_augment", type=int, default=1)
    parser.add_argument("--use_cycle_counter", type=int, default=1)
    parser.add_argument("--train_percentage", type=float, default=0.5)
    parser.add_argument("--no_covariates",
                        action=argparse.BooleanOptionalAction)

    ret_args = parser.parse_args()
    return ret_args


args = get_args()
#%%

config = configparser.ConfigParser()
config.read("../config.ini")

args.hidden_size_lstm = (args.hidden_size_lstm
                         if args.hidden_size_lstm != -1 else args.hidden_size)
save_path = config["PATHS"]["model_path_snl"]
if not os.path.exists(save_path):
    os.makedirs(save_path)

#%%
x, y, c, var = snl_data.load_data()
# c = c[:, :2]

train_idxs, val_idxs, test_idxs = severson_data.get_split(len(x), seed=0)

qc_variance_scaler = StandardScaler().fit(var[train_idxs])

var = qc_variance_scaler.transform(var)

augmented_data = np.hstack([c, var])
x = snl_data.scale_x(x, )

x = snl_data.remove_outliers(x, y)
old_x = x.copy()

smoothed_x = severson_data.smooth_x(x, y, num_points=20)

train_x, train_y, train_s = severson_data.assemble_dataset(
    smoothed_x[train_idxs],
    y[train_idxs] - 10,
    augmented_data[train_idxs],
    seq_len=100,
    use_cycle_counter=args.use_cycle_counter,
)

val_x, val_y, val_s = severson_data.assemble_dataset(
    smoothed_x[val_idxs],
    y[val_idxs],
    augmented_data[val_idxs],
    seq_len=100,
    use_cycle_counter=args.use_cycle_counter,
)

#%%
min_val = 0.85
max_val = 1.0
capacity_output_scaler = MinMaxScaler(
    (-1, 1),
    clip=True).fit(np.maximum(np.minimum(train_y[:, 0:1], max_val), min_val))

train_y[:, 0:1] = capacity_output_scaler.transform(train_y[:, 0:1])

val_y[:, 0:1] = capacity_output_scaler.transform(val_y[:, 0:1])

torch.manual_seed(args.seed)
train_dataset = TensorDataset(
    *[torch.Tensor(input) for input in [train_x, train_y, train_s]])
train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)

val_dataset = TensorDataset(
    *[torch.Tensor(input) for input in [val_x, val_y, val_s]])
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

#%%

input_dim = train_x.shape[
    2]  # Number of input features (e.g. discharge capacity)
num_augment = train_s.shape[1]

if args.no_covariates:
    model = models.Uncertain_LSTM_NoCovariate(
        num_in=input_dim,
        num_augment=num_augment,
        num_hidden=args.hidden_size,
        num_hidden_lstm=args.hidden_size_lstm,
        seq_len=args.sequence_length,
        n_layers=2,
    ).to(device)
else:
    model = models.Uncertain_LSTM(
        num_in=input_dim,
        num_augment=num_augment,
        num_hidden=args.hidden_size,
        num_hidden_lstm=args.hidden_size_lstm,
        seq_len=100,
        n_layers=2,
    ).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

training_loss = []
validation_loss = []

best_val_loss = 500000

cur_patience = 0
max_patience = 5
patience_delta = 0.0
best_weights = None

#%%

for epoch in range(args.num_epochs):

    model.train()
    tr_loss = 0

    for batch_idx, (
            input_data,
            y_hat,
            supp_data,
    ) in enumerate(train_loader):
        model.reset_hidden_state()
        input_data = input_data.to(device)
        supp_data = supp_data.to(device)
        y_hat = y_hat.to(device)
        optimizer.zero_grad()
        (state_mean, state_var) = model(input_data, supp_data)

        # loss
        loss_state = nll_loss(y_hat[:, 0], state_mean[:, 0], state_var[:, 0])
        loss = loss_state
        (loss).backward()
        tr_loss += loss.item()
        optimizer.step()

    tr_loss /= len(train_loader.dataset)
    training_loss.append(tr_loss)

    model.eval()
    val_loss = 0
    val_loss_state = 0
    val_loss_lifetime = 0

    with torch.no_grad():
        for batch_idx, (
                input_data,
                y_hat,
                supp_data,
        ) in enumerate(val_loader):
            model.reset_hidden_state()
            input_data = input_data.to(device)
            supp_data = supp_data.to(device)
            y_hat = y_hat.to(device)

            (state_mean, state_var) = model(input_data, supp_data)
            print

            loss_state = nll_loss(y_hat[:, 0], state_mean[:, 0], state_var[:,
                                                                           0])
            loss = loss_state

            val_loss += loss.item()
            val_loss_state += loss_state.item()

    val_loss /= len(val_loader.dataset)
    val_loss_state /= len(val_loader.dataset)

    val_loss_lifetime /= len(val_loader.dataset)
    validation_loss.append(val_loss)

    print("Epoch: %d, Training loss: %1.5f, Validation loss: %1.5f, " % (
        epoch + 1,
        tr_loss,
        val_loss,
    ))

    if val_loss < best_val_loss:
        best_weights = deepcopy(model.state_dict())
        cur_patience = 0
        best_val_loss = val_loss
    else:
        cur_patience += 1
    if cur_patience > max_patience:
        break

#%%

np.random.seed()
file_name = "".join([str(np.random.choice(10)) for x in range(10)])

results = {}
for arg in vars(args):
    if arg != "save_path":
        results[str(arg)] = getattr(args, arg)

results["train_losses"] = training_loss
results["val_losses"] = validation_loss

model.load_state_dict(best_weights)
model.eval()

results["file_name"] = file_name
results["best_val_loss"] = best_val_loss

pkl.dump(results, open(os.path.join(save_path, file_name + ".pkl"), "wb"))

torch.save(model.state_dict(), oj(save_path, file_name + ".pt"))
print("Saved")
