import inspect
import yaml
from pathlib import Path
import torch
import torch.nn as nn

def make_convs(input_shape, conv_config):
    convs = []
    for i, layer in enumerate(conv_config):
        if layer[0] == "Conv2d":
            if layer[1] == "auto":
                convs.append(NN_MAP[layer[0]](input_shape[0], layer[2], **layer[3]))
            else:
                convs.append(NN_MAP[layer[0]](layer[1], layer[2], **layer[3]))
        elif layer[0] == "MaxPool2d":
            convs.append(NN_MAP[layer[0]](**layer[1]))
        else:
            convs.append(NN_MAP[layer]())

    return nn.Sequential(*convs)


def make_fc(input_dim, out_dim, fc_config):
    fc = []
    for i, layer in enumerate(fc_config):
        if layer[0] == "Linear":
            if layer[1] == "auto" and layer[2] == "auto":
                fc.append(NN_MAP[layer[0]](input_dim, out_dim))
            elif layer[1] == "auto":
                fc.append(NN_MAP[layer[0]](input_dim, layer[2]))
            elif layer[2] == "auto":
                fc.append(NN_MAP[layer[0]](layer[1], out_dim))
            else:
                fc.append(NN_MAP[layer[0]](layer[1], layer[2]))
        else:
            fc.append(NN_MAP[layer]())

    return nn.Sequential(*fc)
