# Compute AlexNet 50 highest singular vectors for every convolutions
import torch

import numpy as np
from collections import Counter

from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method

n_sv = 500

def spec_mnist(self, input, output):
    print(self)
    if is_convolution_or_linear(self):
        s, u, v = k_generic_power_method(self.forward, self.input_sizes[0],
                n_sv,
                max_iter=500,
                use_cuda=torch.cuda.is_available())
        self.spectral_norm = s
        self.u = u
        self.v = v

def save_singular(model, dest_dir):
    layer_names = Counter()
    for layer in model.layers:
        layer_name  = layer._get_name() + "_" + str(layer_names[layer._get_name()])
        layer_names[layer._get_name()] += 1
        output_name = dest_dir + "/" + model._get_name() + "_" + layer_name
        torch.save(layer.spectral_norm, open(output_name + "_spectral", 'wb'))
        torch.save(layer.u, open(output_name + "_left_singular", 'wb'))
        torch.save(layer.v, open(output_name + "_left_singular", 'wb'))

def load_singular(model, dest_dir):
    layer_names = Counter()
    for layer in model.layers:
        layer_name  = layer._get_name() + "_" + str(layer_names[layer._get_name()])
        layer_names[layer._get_name()] += 1
        output_name = dest_dir + "/" + model._get_name() + "_" + layer_name
        torch.load(layer.spectral_norm, open(output_name + "_spectral", 'wb'))
        torch.load(layer.u, open(output_name + "_left_singular", 'wb'))
        torch.load(layer.v, open(output_name + "_left_singular", 'wb'))



