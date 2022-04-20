# Compute AlexNet 50 highest singular vectors for every convolutions
import torch

import numpy as np
from collections import Counter

from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method

n_sv = 200

MAX_ITER = 500

def spec_mnist(self, input, output):
    print(self)
    if is_convolution_or_linear(self):
        if torch.cuda.is_available():
            self.cuda()

        s, u, v = k_generic_power_method(self.forward, self.input_sizes[0],
                n_sv,
                max_iter=MAX_ITER,
                use_cuda=torch.cuda.is_available())
        self.spectral_norm = s
        self.u = u
        self.v = v
    else:
        self.spectral_norm = None
        self.u = None
        self.v = None 

def save_singular(model, dest_dir, model_name):
    layer_names = Counter()
    for layer in model.layers:
        if "Linear" in layer._get_name():
            layer_names = layer_write_singular(layer, dest_dir, model_name, layer_names)
        else:
            for idx in range(len(layer)):
                if is_convolution_or_linear(layer[idx]):
                    layer_names = layer_write_singular(layer[idx], dest_dir, model_name, layer_names)

def layer_write_singular(layer, dest_dir, model_name, layer_names):
    layer_name  = layer._get_name() + "_" + str(layer_names[layer._get_name()])
    layer_names[layer._get_name()] += 1
    output_name = dest_dir + model_name + "_" + layer_name
    print("(Model: ", model_name, ") Name: ", layer_name, ", stored: ", output_name)
    torch.save(layer.spectral_norm, open(output_name + "_spectral", 'wb'))
    torch.save(layer.u, open(output_name + "_left_singular", 'wb'))
    torch.save(layer.v, open(output_name + "_right_singular", 'wb'))
    return layer_names

def load_singular(model, dest_dir, model_name):
    # TODO(as) need to add indexing here to handle nested layers (see save_singular)
    layer_names = Counter()
    for layer in model.layers:
        layer_name  = layer._get_name() + "_" + str(layer_names[layer._get_name()])
        layer_names[layer._get_name()] += 1
        output_name = dest_dir + "/" + model_name + "_" + layer_name
        torch.load(layer.spectral_norm, open(output_name + "_spectral", 'wb'))
        torch.load(layer.u, open(output_name + "_left_singular", 'wb'))
        torch.load(layer.v, open(output_name + "_left_singular", 'wb'))



