import configparser
import pickle
import re
from os.path import join as oj

import numpy as np
from scipy import interpolate

config = configparser.ConfigParser()
config.read("../config.ini")


def get_split(my_len, seed=42):
    """ split a given number of samples reproducibly into training, validation and test

    Args:
        my_len (int]): number of samples to split
        seed (int, optional): [description]. Defaults to 42.

    Returns:
        [type]: tuple of indexes to allocate reproducibly to 
    """

    split_idxs = np.arange(my_len)
    np.random.seed(seed)
    np.random.shuffle(split_idxs)

    num_train, num_val, = (
        int(my_len * 0.5),
        int(my_len * 0.25),
    )
    num_test = my_len - num_train - num_val
    train_idxs = split_idxs[:num_train]
    val_idxs = split_idxs[num_train:num_train + num_val]
    test_idxs = split_idxs[-num_test:]
    return train_idxs, val_idxs, test_idxs


def get_qefficiency(battery, cycle):
    """for a given battery data and cycle, calculate the coloumbic efficiency

    Args:
        battery ([type]): [description]
        cycle ([type]): [description]

    Returns:
        [type]: [description]
    """
    _, cq_curve = get_capacity_curve(battery, cycle, is_discharge=False)
    dv_curve, dq_curve = get_capacity_curve(battery, cycle, is_discharge=True)
    dv_curve, dq_curve = dv_curve[dv_curve > 2.005], dq_curve[dv_curve > 2.005]
    dq_curve_abs = dq_curve.max()  # - dq_curve.min()
    cv_curve_abs = cq_curve.max()  # - cq_curve.min()
    # return dq_curve_abs, cv_curve_abs, dq_curve_abs / cv_curve_abs

    return -1, -1, dq_curve_abs / cv_curve_abs


def get_mean_voltage_difference(battery, cycle):
    """calculate overpotential for a given battery and cycle

    """
    cv_curve, _ = get_capacity_curve(battery, cycle, is_discharge=False)
    dv_curve, dq_curve = get_capacity_curve(battery, cycle, is_discharge=True)
    dv_curve, dq_curve = dv_curve[dv_curve > 2.005], dq_curve[dv_curve > 2.005]

    # start_cap = cq_curve.max()

    return cv_curve.mean() - dv_curve.mean()


def get_capacity_curve(cell, cycle, is_discharge):
    """used to calculate the variance between two cycels, simply returns the relevant parts (ie discharging) of the v and q curve"""

    v_curve = cell["cycles"][str(cycle)]["V"]
    qc_curve = cell["cycles"][str(cycle)]["Qc"]
    qd_curve = cell["cycles"][str(cycle)]["Qd"]

    if qc_curve.max() > 1.1:  # XXX why do this?
        limit_idx = (qc_curve > 1.1).argmax()

    else:
        limit_idx = v_curve.shape[0]

    v_curve = v_curve[:limit_idx]

    if is_discharge:

        q_curve = qd_curve[:limit_idx]

        start = (q_curve > 1e-3).argmax()
        stop = q_curve.argmax()

    else:

        q_curve = qc_curve[:limit_idx]
        start = q_curve.argmin()
        stop = q_curve.argmax()
    return (v_curve[start:stop], q_curve[start:stop])


def smooth_x(
    x,
    y,
    num_points=10,
):
    """Smoothes input for training by averaging over the num_points points.

    Args:
        x Numpy Array: Capacity curves for all batteries
        y ([type]): 1D array indicating the lif
        num_points (int, optional): number of points over which to average. Defaults to 10.

    Returns:
        Numpy Array: [description]
    """
    x_smoothed = x.copy()

    for i in range(len(x)):
        x_smoothed[i, num_points // 2:-num_points // 2 + 1] = np.convolve(
            x[i], np.ones((num_points)) / num_points, mode="valid")
    return x_smoothed


def get_capacity_spline(cell, cycle):
    """
    splines the voltage capacity curve
    """

    v_curve, q_curve = get_capacity_curve(cell, cycle, is_discharge=True)
    f = interpolate.interp1d(v_curve, q_curve, fill_value="extrapolate")
    points = np.linspace(3.5, 2, num=1000)
    spline = f(points)
    spline[np.where(np.isnan(spline))] = 0
    return spline


def scale_x(x, y):
    max_val = 1.1  # nominal capacity for the Severson paper
    end_of_life_val = (
        0.8 * 1.1
    )  # batteries are considered dead after 80%. This should be .8*1.1
    x = np.minimum(x, max_val)
    x = np.maximum(x, end_of_life_val)

    x = (x - end_of_life_val) / (max_val - end_of_life_val)

    return x


def remove_outliers(x_in, y):
    x = x_in.copy()

    for i in range(2, x.shape[1]):
        avg = (x[:, i - 1] + x[:, i - 2]) / 2
        too_low = (x[:, i]) / (avg + 0.0001) < 0.80
        too_high = (x[:, i]) / (avg + 0.0001) > (1.10)
        idx = np.where((too_low + too_high) * (i < y))
        x[idx, i] = x[idx, i - 1]
    return x


def load_data(data_path):
    bat_dicts = load_data_single(data_path)
    data_dict = {}
    for d in bat_dicts[:2]:
        data_dict.update(d)

    return data_dict


def load_data_single(data_path):

    batch1 = pickle.load(open(oj(data_path, "batch1.pkl"), "rb"))
    # remove batteries that do not reach 80% capacity
    del batch1["b1c8"]
    del batch1["b1c10"]
    del batch1["b1c12"]
    del batch1["b1c13"]
    del batch1["b1c22"]
    batch2 = pickle.load(open(oj(data_path, "batch2.pkl"), "rb"))
    # There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
    # and put it with the correct cell from batch1
    batch2_keys = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
    batch1_keys = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
    add_len = [662, 981, 1060, 208, 482]
    for i, bk in enumerate(batch1_keys):
        batch1[bk]["cycle_life"] = batch1[bk]["cycle_life"] + add_len[i]
        for j in batch1[bk]["summary"].keys():
            if j == "cycle":
                batch1[bk]["summary"][j] = np.hstack((
                    batch1[bk]["summary"][j],
                    batch2[batch2_keys[i]]["summary"][j] +
                    len(batch1[bk]["summary"][j]),
                ))
            else:
                batch1[bk]["summary"][j] = np.hstack(
                    (batch1[bk]["summary"][j],
                     batch2[batch2_keys[i]]["summary"][j]))
        last_cycle = len(batch1[bk]["cycles"].keys())
        for j, jk in enumerate(batch2[batch2_keys[i]]["cycles"].keys()):
            batch1[bk]["cycles"][str(last_cycle +
                                     j)] = batch2[batch2_keys[i]]["cycles"][jk]
    del batch2["b2c7"]
    del batch2["b2c8"]
    del batch2["b2c9"]
    del batch2["b2c15"]
    del batch2["b2c16"]

    batch3 = pickle.load(open(oj(data_path, "batch3.pkl"), "rb"))
    # remove noisy channels from batch3
    del batch3["b3c37"]
    del batch3["b3c2"]
    del batch3["b3c23"]
    del batch3["b3c32"]
    del batch3["b3c38"]
    del batch3["b3c39"]

    return [batch1, batch2, batch3]


def get_charge_policy(my_string):

    # charge policy extract from string
    vals = [
        float(x[0] + x[1]) for x in re.findall(r"(\d\.\d+)|(\d+)", my_string)
    ]  # get three  ints or floats from a string,
    return vals


def get_max_life_time(data_dict):

    max_lifetime = 0
    for bat in data_dict.keys():
        max_lifetime = np.maximum(max_lifetime,
                                  data_dict[bat]["cycle_life"][0][0])
    return int(max_lifetime)


def transform_charging(c):
    num_steps = 9
    stop_val = 0.8
    long_c = np.zeros((len(c), num_steps))
    for i in range(num_steps):
        use_first = c[:, -1] > i / num_steps * stop_val
        long_c[:, i] = c[:, 0] * use_first + c[:, 2] * (1 - use_first)
    return long_c


def get_elis_data(
    data_dict,
    num_offset=0,
):

    max_lifetime = get_max_life_time(data_dict) - num_offset
    num_bats = len(data_dict)
    coloumbic_eff = -1 * np.ones((num_bats, max_lifetime))
    list_of_keys = []

    x = -1 * np.ones((num_bats, max_lifetime))
    charge_policy = np.zeros((num_bats, 4))

    y = np.zeros(num_bats)
    err_id = 0
    for i, bat in enumerate(data_dict.keys()):
        list_of_keys.append(bat)
        first, switch_time, second = get_charge_policy(
            data_dict[bat]["charge_policy"])

        switch_time /= 100

        avg = first * (switch_time / 0.8) + second * (
            1 - switch_time / 0.8)  # batteries charged from .0 until .8 SOC
        charge_policy[i] = first, avg, second, switch_time
        for j in range(0, len(data_dict[bat]["summary"]["QC"]) - num_offset):
            bat_val = data_dict[bat]

            try:
                coloumbic_eff[i, j] = get_qefficiency(bat_val, j)[2]
            except ValueError:
                err_id += 1

            pass

        x[i, :len(data_dict[bat]["summary"]["QC"]) -
          num_offset] = data_dict[bat]["summary"]["QC"][num_offset:]
        y[i] = data_dict[bat]["cycle_life"]
    print(err_id)
    y = y.astype(np.int32)
    x[:41, :-1] = x[:41, 1:]

    return x, y, coloumbic_eff, list_of_keys, charge_policy


def get_poul_data(
    data_dict,
    num_offset=0,
):
    """
    for the meeting with Poul and subsequent notebook creation on aged data. Not needed for anything else


    """

    max_lifetime = get_max_life_time(data_dict) - num_offset
    num_bats = len(data_dict)
    coloumbic_eff = -1 * np.ones((num_bats, max_lifetime))
    overpotencial = -1 * np.ones((num_bats, max_lifetime))
    list_of_keys = []

    x = -1 * np.ones((num_bats, max_lifetime))
    x_discharge = -1 * np.ones((num_bats, max_lifetime))
    charge_policy = np.zeros((num_bats, 4))

    y = np.zeros(num_bats)
    err_id = 0
    for i, bat in enumerate(data_dict.keys()):
        list_of_keys.append(bat)
        first, switch_time, second = get_charge_policy(
            data_dict[bat]["charge_policy"])

        switch_time /= 100

        avg = first * (switch_time / 0.8) + second * (
            1 - switch_time / 0.8)  # batteries charged from .0 until .8 SOC
        charge_policy[i] = first, avg, second, switch_time
        for j in range(0, len(data_dict[bat]["summary"]["QC"]) - num_offset):
            bat_val = data_dict[bat]

            try:
                coloumbic_eff[i, j] = get_qefficiency(bat_val, j)[2]
                overpotencial[i, j] = get_mean_voltage_difference(bat_val, j)

            except ValueError:
                err_id += 1

            pass

        x[i, :len(data_dict[bat]["summary"]["QC"]) -
          num_offset] = data_dict[bat]["summary"]["QC"][num_offset:]
        x_discharge[i, :len(data_dict[bat]["summary"]["QC"]) -
                    num_offset] = data_dict[bat]["summary"]["QD"][num_offset:]
        y[i] = data_dict[bat]["cycle_life"]
    print(err_id)
    y = y.astype(np.int32)
    x[:41, :-1] = x[:41, 1:]

    return x, y, coloumbic_eff, list_of_keys, charge_policy, x_discharge, overpotencial


def get_capacity_input(data_dict,
                       num_offset=0,
                       start_cycle=10,
                       stop_cycle=100,
                       use_long_cschedule=False):
    for key in data_dict.keys():
        bat = data_dict[key]

        start_curve = get_capacity_spline(
            bat,
            start_cycle,
        )
        stop_curve = get_capacity_spline(
            bat,
            stop_cycle,
        )
        idxs = np.where(
            1 - (np.isnan(start_curve) + np.isnan(stop_curve))
        )  # the first parts of the cycle don't get modelled well. Ignored because tiny part
        bat["qv_variance"] = np.log(
            (start_curve[idxs] - stop_curve[idxs]).var())
        bat["c_efficiency"] = (get_qefficiency(bat, stop_cycle)[2] -
                               get_qefficiency(bat, start_cycle)[2])
        bat["v_delta"] = get_mean_voltage_difference(
            bat, stop_cycle) - get_mean_voltage_difference(
                bat, start_cycle)  # always use the fifth cycle

    max_lifetime = get_max_life_time(data_dict) - num_offset

    num_bats = len(data_dict)
    x = -1 * np.ones((num_bats, max_lifetime))
    charge_policy = np.zeros((num_bats, 4))
    in_cycle_data = np.zeros((num_bats, 3))

    y = np.zeros(num_bats)
    for i, bat in enumerate(data_dict.keys()):
        x[i, :len(data_dict[bat]["summary"]["QC"]) -
          num_offset] = data_dict[bat]["summary"]["QC"][num_offset:]
        y[i] = data_dict[bat]["cycle_life"]
        first, switch_time, second = get_charge_policy(
            data_dict[bat]["charge_policy"])

        switch_time /= 100

        avg = first * (switch_time / 0.8) + second * (
            1 - switch_time / 0.8)  # batteries charged from .0 until .8 SOC
        charge_policy[i] = first, avg, second, switch_time
        in_cycle_data[i, 0] = data_dict[bat]["qv_variance"]
        in_cycle_data[i, 1] = data_dict[bat]["c_efficiency"]
        in_cycle_data[i, 2] = data_dict[bat]["v_delta"]
    y = y.astype(np.int32)
    x[:41, :
      -1] = x[:41,
              1:]  #  in severson data the first batch starts with 0, likely a bug
    if use_long_cschedule:
        return x, y, charge_policy, in_cycle_data

    return x, y, charge_policy[:, :3], in_cycle_data


def assemble_dataset(x, y, augment, seq_len=50, use_cycle_counter=True):

    xs = []
    ys = []
    add_data_s = []
    for i in range(len(x)):
        for j in range(y[i] - seq_len - 3):

            xs.append(x[i, j:j + seq_len])

            ratio = x[i, j + seq_len] / (x[i, j + seq_len - 1] + 10e-17)

            ys.append((ratio, (y[i] - seq_len - j)))
            if use_cycle_counter:
                add_data_s.append(np.hstack([augment[i], np.log(j + seq_len)]))
            else:
                add_data_s.append(augment[i])
    return np.asarray(xs)[:, :, None], np.asarray(ys), np.asarray(add_data_s)
