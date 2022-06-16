import configparser

import torch

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

config = configparser.ConfigParser()
config.read("../config.ini")



def nll_loss(target, mean, var_log):

    nll = 0.5 * (torch.log(var_log) + torch.square(target - mean) / ((var_log)))
    return torch.sum(nll)


def mse_loss(target, mean, var_log):

    return torch.sum(torch.square(target - mean))


def l1_loss(target, mean, var_log):

    # nll=  var_log/2 + torch.square(target -mean)/(2 * torch.exp(var_log))
    return torch.sum(torch.abs(target - mean))


def elastic_loss(target, mean, y):
    return l1_loss_entire_curve(target, mean, y) + l2_loss_entire_curve(target, mean, y)


def l1_loss_entire_curve(
    target,
    mean,
    y,
):

    # make mask
    # mask = torch.zeros_like(target, dtype = torch.bool)
    # for i,j in enumerate(range(len(y))):
    #     mask[i, :np.minimum(2237,j+50)] = 1
    # return torch.masked_select(torch.abs(target -mean), mask).sum()
    return torch.abs(target - mean).sum()


def l2_loss_entire_curve(
    target,
    mean,
    y,
):

    # # make mask
    # mask = torch.zeros_like(target, dtype = torch.bool)

    # for i,j in enumerate(range(len(y))):
    #     mask[i, :np.minimum(2237,j+50)] = 1
    # return torch.masked_select(torch.square(target -mean), mask).sum()
    return torch.square(target - mean).sum()
