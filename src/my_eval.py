# print(__file__)
import sys

import torch
import torch.utils.data

sys.path.insert(0, "src_lstm")

import numpy as np

cutoff_val = 10e-2


def get_rmse_transfer(
    model,
    idxs,
    x,
    y,
    augmented_data,
    charging_sched,
    seq_length,
    device,
    scaler,
    use_cycle_counter=True,
    max_steps=3500,
):

    test_seq = x[idxs][:, :seq_length, None].copy()
    rmse_lifetime = 0

    with torch.no_grad():

        for i in range(max_steps - seq_length):
            if use_cycle_counter:
                supp_val_data = np.hstack(
                    [
                        augmented_data[idxs],
                        charging_sched[idxs, :, i],
                        np.ones((len(idxs), 1)) * np.log(i + seq_length),
                    ]
                )
            else:
                supp_val_data = np.hstack(
                    [
                        augmented_data[idxs],
                        charging_sched[idxs, :, i],
                    ]
                )
            supp_val_data_torch = torch.from_numpy(supp_val_data).to(device).float()
            test_seq_torch = (
                torch.from_numpy(test_seq[:, -seq_length:]).to(device).float()
            )
            model.reset_hidden_state()
            (pred_state, _), (pred_lifetime, _) = model(
                test_seq_torch, supp_val_data_torch
            )
            if i == 0:
                rmse_lifetime = np.sqrt(
                    np.square(
                        torch.exp(pred_lifetime).cpu().numpy()[:, 0]
                        + seq_length
                        - y[idxs]
                    ).mean()
                )
            pred_state = torch.clip(
                torch.from_numpy(scaler.inverse_transform(pred_state.cpu().numpy())).to(
                    device
                ),
                0,
                1,
            )
            pred_state = pred_state[:, 0] * test_seq_torch[:, -1, 0]
            test_seq = np.hstack([test_seq, pred_state.cpu().numpy()[:, None, None]])

        y_lifetime_pred_state = (test_seq[:, :, 0] < cutoff_val).argmax(
            axis=1
        )  # 0 index is because this is only one output
    return np.sqrt((np.square(y[idxs] - y_lifetime_pred_state)).mean()), rmse_lifetime


# def get_predictions(
#     model,
#     idxs,
#     x,
#     y,
#     augmented_data,
#     seq_length,
#     device,
#     scaler,
#     use_cycle_counter=True,
#     num_steps=5000,
# ):
#     if use_cycle_counter:

#         supp_val_data = np.hstack(
#             [augmented_data[idxs], np.ones((len(idxs), 1)) * np.log(seq_length)]
#         )
#     else:
#         supp_val_data = augmented_data[idxs]

#     test_seq = x[idxs][:, :seq_length, None].copy()

#     with torch.no_grad():

#         for i in range(num_steps - seq_length):
#             supp_val_data_torch = torch.from_numpy(supp_val_data).to(device).float()
#             test_seq_torch = (
#                 torch.from_numpy(test_seq[:, -seq_length:]).to(device).float()
#             )
#             model.reset_hidden_state()
#             (pred_state, _) = model(test_seq_torch, supp_val_data_torch)
#             # if i ==0:
#             #     first_life_pred = torch.exp(pred_lifetime).cpu().numpy()[:,0].copy()  + seq_length
#             # rmse_lifetime = np.sqrt(np.square(torch.exp(pred_lifetime).cpu().numpy()[:,0]  -y[idxs]).mean())

#             pred_state = torch.clip(
#                 torch.from_numpy(scaler.inverse_transform(pred_state.cpu().numpy())).to(
#                     device
#                 ),
#                 0,
#                 1,
#             )
#             pred_state = pred_state[:, 0] * test_seq_torch[:, -1, 0]
#             test_seq = np.hstack([test_seq, pred_state.cpu().numpy()[:, None, None]])
#             if use_cycle_counter:
#                 supp_val_data[:, -1] = np.log(np.exp(supp_val_data[:, -1]) + 1)
#         y_lifetime_pred_state = (test_seq[:, :, 0] < cutoff_val).argmax(axis=1)
#         # y_lifetime_pred[y_lifetime_pred ==0 ]= y.max()
#     return (y_lifetime_pred_state,)


def get_rmse_dnn(
    model,
    idxs,
    x,
    y,
    device,
):

    with torch.no_grad():
        pred_curves = model(torch.from_numpy(x[idxs]).to(device).float())

        pred_curves = pred_curves.cpu().numpy()
        y_lifetime_pred = (
            pred_curves[
                :,
                :,
            ]
            < cutoff_val
        ).argmax(axis=1)

    return np.sqrt((np.square(y[idxs] - y_lifetime_pred)).mean())


def get_rmse(
    model,
    idxs,
    x,
    y,
    augmented_data,
    seq_length,
    device,
    scaler,
    max_steps=5000,
    use_cycle_counter=True,
):

    if use_cycle_counter:

        supp_val_data = np.hstack(
            [augmented_data[idxs], np.ones((len(idxs), 1)) * np.log(seq_length)]
        )
    else:
        supp_val_data = augmented_data[idxs]

    test_seq = x[idxs][:, :seq_length, None].copy()

    with torch.no_grad():
        while (np.all(test_seq[:, -1] < 1e-3) == False) * (
            test_seq.shape[1] < max_steps
        ):
            # for i in range(max_steps - seq_length):
            supp_val_data_torch = torch.from_numpy(supp_val_data).to(device).float()
            test_seq_torch = (
                torch.from_numpy(test_seq[:, -seq_length:]).to(device).float()
            )
            model.reset_hidden_state()
            (pred_state, _) = model(test_seq_torch, supp_val_data_torch)

            pred_state = torch.from_numpy(
                scaler.inverse_transform(pred_state.cpu().numpy())
            ).to(device)

            # ((x[i, j + seq_len - 5: j + seq_len ]).mean() + 10e-17)
            pred_state = pred_state[:, 0] * (test_seq_torch[:, -1, 0] + 10e-17)
            test_seq = np.hstack([test_seq, pred_state.cpu().numpy()[:, None, None]])
            if use_cycle_counter:
                supp_val_data[:, -1] = np.log(np.exp(supp_val_data[:, -1]) + 1)

        y_lifetime_pred = (test_seq[:, :, 0] < cutoff_val).argmax(axis=1)

    return np.sqrt((np.square(y[idxs] - y_lifetime_pred)).mean())
