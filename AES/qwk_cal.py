import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def qwk_calculation(y_true, y_pred):
    if isinstance(y_pred, torch.Tensor):
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

    if torch.all(y_true == 0) and torch.all(y_pred == 0):
        return torch.tensor([1])

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")



if __name__ == '__main__':
    a = torch.zeros((1783, 1))
    b = torch.tensor([1783])
    print(a*b)
