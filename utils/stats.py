"""Utils for Statistics"""

from timeit import default_timer as timer
import torch
from scipy.stats import entropy
from scipy.special import softmax
import numpy as np


def uncertainty(logits):
    return entropy(softmax(logits, axis=-1), axis=-1)


def drift_statistics(dataloader, model, drift_detector, device):
    accs = []
    times = []
    drift_pos = []
    uncertainties = np.asarray([])
    start = timer()
    for inputs, y in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs).cpu()
        torch.cuda.empty_cache()
        result = drift_detector.get_result(outputs)
        times.append(timer())
        accs.append((outputs.max(1)[1] == y).float().sum()/len(y))
        drift_pos.append(result['is_drift'])
        uncertainties = np.concatenate((uncertainties, uncertainty(outputs)))
    return accs, drift_pos, uncertainties, times, start


def confusion_matrix(accs, drift_pos, threshold):
    """
    Return false positive rate and true positive rate
    """
    t_p, f_p, f_n, t_n = 0, 0, 0, 0
    
    for domain_idx in accs:

        accuracy = accs[domain_idx]
        drift = drift_pos[domain_idx]
        l = len(accuracy)
        accuracy = [1 if i > threshold else 0 for i in accuracy]
        t_p += sum([1 if (accuracy[i] < threshold and  drift[i] == 1) else 0 for i in range(0,l)])
        f_p += sum([1 if (accuracy[i] > threshold and  drift[i] == 1) else 0 for i in range(0,l)])
        f_n += sum([1 if (accuracy[i] < threshold and  drift[i] == 0) else 0 for i in range(0,l)])
        t_n += sum([1 if (accuracy[i] > threshold and  drift[i] == 0) else 0 for i in range(0,l)])
        
    FPR = f_p if f_p + t_n == 0 else f_p / (f_p + t_n)
    TPR = t_p if t_p + f_n == 0 else t_p / (t_p + f_n)
    
    return FPR, TPR







