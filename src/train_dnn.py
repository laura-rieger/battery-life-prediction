import os
import sys

#
# cur_path = pathlib.Path(__file__).parent.absolute()
# os.chdir(cur_path)

sys.path.insert(0, "src_lstm")
import configparser
import pickle as pkl
import sys
from argparse import ArgumentParser
from copy import deepcopy
from os.path import join as oj

import numpy as np
import torch
import torch.utils.data
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import loss_functions
import models
import my_eval
import severson_data

# torch.backends.cudnn.enabled = False
cuda = torch.cuda.is_available()

if not cuda:
    sys.exit()
# torch.set_num_threads(1)
device = torch.device("cuda")

#%%


def get_args():
    parser = ArgumentParser(description="Battery cycle prediction with LSTM")
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path",
                        type=str,
                        default="../models/dnn_models")
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--experiment_name", type=str, default="DNN")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bootstrap", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--train_percentage", type=float, default=1)
    parser.add_argument("--sequence_length", type=int, default=100)

    ret_args = parser.parse_args()
    return ret_args


args = get_args()
#%%

config = configparser.ConfigParser()
config.read("../config.ini")

save_path = config["PATHS"]["model_path_severson"]
if not os.path.exists(save_path):
    os.makedirs(save_path)

#%%
data_path = config["DATASET"]["severson_path"]
data_dict = severson_data.load_data(data_path)

#%%
x, y, c, var = severson_data.get_capacity_input(
    data_dict,
    num_offset=0,
    start_cycle=args.start,
    stop_cycle=args.sequence_length,
)
x = severson_data.scale_x(x, y)
x = severson_data.remove_outliers(x, y)
x[np.where(x == -1)] = 0
old_x = x.copy()

#%%

train_idxs, val_idxs, test_idxs = severson_data.get_split(len(x), seed=42)

if args.train_percentage != 1:
    train_idxs = train_idxs[:int(args.train_percentage * len(train_idxs))]

qc_variance_scaler = StandardScaler().fit(var[train_idxs])
var = qc_variance_scaler.transform(var)

#%%

augmented_data = np.hstack([c, var, x[:, :1]])

#%%

torch.manual_seed(args.seed)
train_dataset = TensorDataset(*[
    torch.Tensor(input)
    for input in [augmented_data[train_idxs], x[train_idxs], y[train_idxs]]
])
val_dataset = TensorDataset(*[
    torch.Tensor(input)
    for input in [augmented_data[val_idxs], x[val_idxs], y[val_idxs]]
])

test_dataset = TensorDataset(*[
    torch.Tensor(input)
    for input in [augmented_data[test_idxs], x[test_idxs], y[test_idxs]]
])

train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False)

#%%

input_dim = augmented_data.shape[1]

model = models.CapacityDNN(
    num_input=input_dim,
    num_hidden=args.hidden_size,
    dropout=args.dropout,
    num_output=2237,
).to(device)

optimizer = optim.Adam(model.parameters(), )

training_loss = []
validation_loss = []

best_val_loss = 500000

cur_patience = 0
max_patience = 3
patience_delta = 0.0
best_weights = None

loss_function = loss_functions.elastic_loss
#%%,


def train(model, epoch, loader, is_train=False):
    if is_train:
        model.train()
    else:
        model.eval()

    accumulated_loss = 0

    for batch_idx, (x_batch, y_batch, eol_batch) in enumerate(loader):

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_batch_pred = model(x_batch)

        loss = loss_function(y_batch, y_batch_pred, eol_batch)

        loss.backward()
        accumulated_loss += loss.item()
        optimizer.step()
    accumulated_loss /= len(train_loader.dataset)
    return accumulated_loss


for epoch in range(args.num_epochs):
    train_loss = train(model, epoch, train_loader, is_train=True)
    training_loss.append(train_loss)

    val_loss = train(model, epoch, val_loader, is_train=False)
    validation_loss.append(val_loss)

    print("Epoch: %d, Training loss: %1.5f, Validation loss: %1.5f, " % (
        epoch + 1,
        train_loss,
        val_loss,
    ))

    if val_loss + patience_delta < best_val_loss:

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
# file_name = "1234"
results = {}
for arg in vars(args):
    if arg != "save_path":
        results[str(arg)] = getattr(args, arg)

results["train_losses"] = training_loss
results["val_losses"] = validation_loss

model.load_state_dict(best_weights)
model.eval()

#%%
results["rmse_state_val"] = my_eval.get_rmse_dnn(model, val_idxs,
                                                 augmented_data, y, device)
results["rmse_state_train"] = my_eval.get_rmse_dnn(model, train_idxs,
                                                   augmented_data, y, device)
results["rmse_state_test"] = my_eval.get_rmse_dnn(model, test_idxs,
                                                  augmented_data, y, device)

results["file_name"] = file_name
results["best_val_loss"] = best_val_loss

pkl.dump(results, open(os.path.join(args.save_path, file_name + ".pkl"), "wb"))

torch.save(model.state_dict(), oj(args.save_path, file_name + ".pt"))
print("Saved")
