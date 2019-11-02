from .imports import *

def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy()/len(original)