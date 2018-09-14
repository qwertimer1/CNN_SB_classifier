import torch.nn.functional as F


def my_loss(y_input, y_target):
    return F.nll_loss(y_input, y_target)

def cross_entorpy(y_input, y_target):
    return nn.cross_entorpy(y_input, y_target)

def other_loss(y_input, y_target):
    """look at implementing other loss funcs here (KL, etc)
    """
