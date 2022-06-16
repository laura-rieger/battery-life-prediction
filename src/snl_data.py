import configparser
import pickle as pkl
from os.path import join as oj
from scipy import interpolate
import numpy as np
import re

config = configparser.ConfigParser()
config.read("../config.ini")


def get_capacity_spline(cell, start_point, stop_point, use_tenth=True):
    """
    splines the voltage capacity curve
    """
    if use_tenth:

        v_curve, q_curve = cell['10th Vdch [V]'], cell['10th Qdch [Ah]']
    else:
        v_curve, q_curve = cell['100th Vdch [V]'], cell['100th Qdch [Ah]']
    f = interpolate.interp1d(v_curve, q_curve, fill_value="extrapolate")
    points = np.linspace(stop_point, start_point, num=1000)
    spline = f(points)
    spline[np.where(np.isnan(spline))] = 0
    return spline


def remove_outliers(x_in, y):
    x = x_in.copy()

    for i in range(2, x.shape[1]):
        avg = (x[:, i - 1] + x[:, i - 2]) / 2
        too_low = (x[:, i]) / (avg + 0.0001) < 0.80
        too_high = (x[:, i]) / (avg + 0.0001) > (1.10)
        too_high = x[:, i] > 1
        idx = np.where((too_low + too_high) * (i < y))
        idx = np.where((too_high) * (i < y))
        x[idx, i] = x[idx, i - 1]
    return x


def scale_x(x, ):
    max_val = x[:, 0].max()  # no idea what the nominal value is
    x /= max_val
    x[x < 0] = -1

    return x


def load_data(include_key_list=False):
    snl_dict = pkl.load(
        open(oj(config['DATASET']['data_path'], 'snl_curated_data.pkl'), "rb"))
    # don't use LFP as we are looking at other chemistries
    snl_dict = {
        key: val
        for key, val in snl_dict.items()
        if 'LFP' != snl_dict[key]['Cathode chemistry']
    }
    key_list = list(snl_dict.keys())
    max_lifetime = max(
        [len(snl_dict[key]['Max Qdischarge [Ah]']) for key in key_list])
    num_bats = len(key_list)

    for key in key_list:
        overpotential_10 = snl_dict[key]['10th Vch [V]'].mean(
        ) - snl_dict[key]['10th Vdch [V]'].mean()
        overpotential_100 = snl_dict[key]['100th Vch [V]'].mean(
        ) - snl_dict[key]['100th Vdch [V]'].mean()
        snl_dict[key]['overpotential'] = overpotential_100 - overpotential_10
        c_eff_10 = snl_dict[key]['10th Qdch [Ah]'].max()
        c_eff_100 = snl_dict[key]['100th Qdch [Ah]'].max()
        snl_dict[key]['c_eff'] = c_eff_100 - c_eff_10
        cur_bat = snl_dict[key]

        min_voltage = np.maximum(cur_bat['100th Vdch [V]'].min(),
                                 cur_bat['10th Vdch [V]'].min())
        max_voltage = np.minimum(cur_bat['100th Vdch [V]'].max(),
                                 cur_bat['10th Vdch [V]'].max())
        start_curve = get_capacity_spline(cur_bat,
                                          min_voltage,
                                          max_voltage,
                                          use_tenth=True)
        stop_curve = get_capacity_spline(cur_bat,
                                         min_voltage,
                                         max_voltage,
                                         use_tenth=False)
        snl_dict[key]['var'] = (start_curve - stop_curve).var()
    x = -np.ones((num_bats, max_lifetime))
    y = np.zeros(num_bats, dtype=np.int32)
    c = np.zeros((num_bats, 2))

    covariates = np.zeros((num_bats, 3))
    for i, key in enumerate(key_list):
        y[i] = len(snl_dict[key]['Max Qdischarge [Ah]'])

        x[i, :y[i]] = snl_dict[key]['Max Qdischarge [Ah]']
        covariates[i, 0] = snl_dict[key]['var']
        covariates[i, 1] = snl_dict[key]['c_eff']
        covariates[i, 2] = snl_dict[key]['overpotential']
        c[i, 0] = snl_dict[key]['Charging rate']
        c[i, 1] = snl_dict[key]['Discharging rate'][:-1]

        # c[i, 2] = re.findall(r'\d+', key)[1]
    if include_key_list:
        return x, y, c, covariates, key_list
    else:
        return x, y, c, covariates
