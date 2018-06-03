import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def set_forget_bias(lstm, num):
    for names in lstm._all_weights:
        for name in filter(lambda n: "bias" in n, names):
            bias = getattr(lstm, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(num)

def lstm_init_uniform_weights(lstm, scale):
    for layer_p in lstm._all_weights:
        for p in layer_p:
            if 'weight' in p:
                init.uniform_(lstm.__getattr__(p), 0.0, scale)

def linear_init(l, scale):
    l.weight.data.uniform_(0.0, scale)
    l.bias.data.fill_(0)
