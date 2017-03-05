## based on https://github.com/dmlc/mxnet/issues/1302
## Parses the model fit log file and generates a train/val vs epoch plot
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_log(log_file,metric='accuracy'):
    TR_RE = re.compile('.*?]\sTrain-'+metric+'=([\d\.]+)')
    VA_RE = re.compile('.*?]\sValidation-'+metric+'=([\d\.]+)')
    log = open(log_file).read()
    train_metric = [float(x) for x in TR_RE.findall(log)]
    validation_metric = [float(x) for x in VA_RE.findall(log)]
    return (train_metric,validation_metric)

def plot_curves(train_metric,validation_metric,metric='accuracy',ylim=[0,1]):
    idx = np.arange(len(train_metric))
    plt.figure(figsize=(8, 6))
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.plot(idx, train_metric, 'o', linestyle='-', color="r",
             label="Train "+metric)

    plt.plot(idx, validation_metric, 'o', linestyle='-', color="b",
             label="Validation "+metric)

    plt.legend(loc="best")
    plt.xticks(np.arange(min(idx), max(idx)+1, 5))
    plt.yticks(np.arange(ylim[0], ylim[1], 0.2))
    plt.ylim(ylim)
    plt.show()