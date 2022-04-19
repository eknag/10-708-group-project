
from copy import deepcopy
import math
import torch
import torchvision

import numpy as np
from collections import Counter

from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method

from seqlip import optim_nn_pca_greedy

from experiments.model_get_sv import compute_module_input_sizes, execute_through_model, save_singular, spec_mnist

# TODO(as) play around with this and see what happens, can go up to 500
n_sv = 200

def get_lipschitz(model, out_dir, model_name, calc_sing=True):  
    # Handle formatting of output directory
    if out_dir[-1] != '/':
        out_dir += '/'

    if calc_sing:
        # Taken from experiments/model_get_sv.py.  Assumes eval() has already been run on the model.
        for p in model.parameters():
            p.requires_grad = False

        # Stores largest singular values inside the model itself
        input_size = model.input_dim
        if len(input_size) == 3:
            input_size = [1, *input_size]
        elif len(input_size) != 4:
            print("ERROR: invalid input image size")
            return -1, -1 
        compute_module_input_sizes(model, input_size)
        execute_through_model(spec_mnist, model)
        
        # Store singular values in files
        save_singular(model, out_dir)

    # Taken from experiments/model.py
    return model_operations(model, out_dir)





def model_operations(model, dest_dir):
    """ Starting point of the script is a saving of all singular values and vectors
    in mnist_save/

    We perform the 100-optimization implemented in optim_nn_pca_greedy
    """
    for p in model.parameters():
        p.requires_grad = False

    # No longer needed since model already has sizes baked in
    # compute_module_input_sizes(model, [1, 1, 28, 28])

    lip = 1
    lip_spectral = 1

    # Determine number of convolution/linear layers
    relevant_layer_cnt = 0
    for layer in model.layers:
        for idx in range(len(layer)):
            if is_convolution_or_linear(layer[idx]):
                relevant_layer_cnt += 1

    # Indices of convolutions and linear layers
    layer_names = Counter()
    conv_lin_idx = 0
    for layer in model.layers:
        for idx in range(len(layer)):    
            if not is_convolution_or_linear(layer[idx]):
                continue

            print('Dealing with {}'.format(layer[idx]._get_name()))

            # Generate file name
            layer_name  = layer[idx]._get_name() + "_" + str(layer_names[layer[idx]._get_name()])
            layer_names[layer[idx]._get_name()] += 1
            output_name = dest_dir + model._get_name() + "_" + layer_name

            # Load from file
            U = torch.load(output_name + "_left_singular")
            U = torch.cat(U[:n_sv], dim=0).view(n_sv, -1)
            su = torch.load(output_name + "_spectral")
            su = su[:n_sv]

            V = torch.load(output_name + "_right_singular")
            V = torch.cat(V[:n_sv], dim=0).view(n_sv, -1)
            sv = torch.load(output_name + "_spectral")
            sv = sv[:n_sv]
            print('Ratio layer i  : {:.4f}'.format(float(su[0] / su[-1])))
            print('Ratio layer i+1: {:.4f}'.format(float(sv[0] / sv[-1]))) 
            U, V = U.cpu(), V.cpu()               

            # Set up
            if conv_lin_idx == 0:
                sigmau = torch.diag(torch.Tensor(su))
            else:
                sigmau = torch.diag(torch.sqrt(torch.Tensor(su)))

            if conv_lin_idx == relevant_layer_cnt - 1:
                sigmav = torch.diag(torch.Tensor(sv))
            else:
                sigmav = torch.diag(torch.sqrt(torch.Tensor(sv)))

            expected = sigmau[0,0] * sigmav[0,0]
            print('Expected: {}'.format(expected))

            lip_spectral *= expected

            # Calculate approximation
            curr, _ = optim_nn_pca_greedy(U.t() @ sigmau, sigmav @ V, use_cuda=torch.cuda.is_available())
            print('Approximation: {}'.format(curr))
            lip *= float(curr)
            conv_lin_idx += 1


    print('Lipschitz spectral: {}'.format(lip_spectral))
    print('Lipschitz approximation: {}'.format(lip))
    return lip_spectral, lip


