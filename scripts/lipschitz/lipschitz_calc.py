
from copy import deepcopy
import math
import torch
import torchvision
import os

import numpy as np
from collections import Counter

from lipschitz.lipschitz_utils import *
from lipschitz.max_eigenvalue import k_generic_power_method

from lipschitz.seqlip import optim_nn_pca_greedy

from lipschitz.model_get_sv import compute_module_input_sizes, execute_through_model, save_singular, spec_mnist

LIP_OUT_SUBDIR = "lipschitz"

# Set to 200 in the original repo, but they train 500 for some reason
n_sv = 200
OPTIM_ITER = 3


def calc_singular_val(model, out_dir, model_name):
    # Handle formatting of output directory
    if out_dir[-1] != '/':
        out_dir += '/'

    # Taken from experiments/model_get_sv.py.  Assumes eval() has already been run on the model.
    for p in model.parameters():
        p.requires_grad = False

    # Stores largest singular values inside the model itself
    if "encoder" in model_name:
        input_size = model.input_dim
    else:
        # decoder input is a vector
        input_size = (1, 1, 1, model.layers[0].in_features)
    if len(input_size) == 3:
        input_size = [1, *input_size]
    elif len(input_size) != 4:
        print("ERROR: invalid input image size")
    compute_module_input_sizes(model, input_size)
    execute_through_model(spec_mnist, model)
    
    # Store singular values in files
    save_singular(model, out_dir, model_name)

def get_lipschitz(model, out_dir, model_name):  
    # Handle formatting of output directory
    if out_dir[-1] != '/':
        out_dir += '/'
    lipschitz_output_dir = os.path.join(out_dir, LIP_OUT_SUBDIR)
    os.makedirs(lipschitz_output_dir, exist_ok=True)
    
    if "encoder" in model_name:
        return -1, -1
        
    # Taken from experiments/model.py
    return model_operations(model, out_dir, lipschitz_output_dir, model_name)


def model_operations(model, source_dir, dest_dir, model_name):
    """ Starting point of the script is a saving of all singular values and vectors
    in mnist_save/

    We perform the 100-optimization implemented in optim_nn_pca_greedy
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False
        if use_cuda:
            p.cuda()

    # No longer needed since model already has sizes baked in
    # compute_module_input_sizes(model, [1, 1, 28, 28])

    lip = 1
    lip_spectral = 1

    # Determine number of convolution/linear layers
    relevant_layer_cnt = 0
    for layer in model.layers:
        if "Linear" in layer._get_name():
            relevant_layer_cnt += 1
        else:
            for idx in range(len(layer)):
                if is_convolution_or_linear(layer[idx]):
                    relevant_layer_cnt += 1


    if dest_dir[-1] != '/':
        dest_dir += '/'
    if source_dir[-1] != '/':
        source_dir += '/'

    # Indices of convolutions and linear layers
    layer_names = Counter()
    conv_lin_idx = 0
    print("Calculating Lipschitz constant for ", model_name)
    for layer in model.layers:
        if "Linear" in layer._get_name():
            # Generate file name
            layer_name  = layer._get_name() + "_" + str(layer_names[layer._get_name()])
            layer_names[layer._get_name()] += 1
            output_name = source_dir + model_name + "_" + layer_name

            # Process layer
            lip_spectral, lip = layer_processing(lip_spectral, lip, layer, output_name, relevant_layer_cnt, conv_lin_idx)
            conv_lin_idx += 1
        else:
            for idx in range(len(layer)):
                if is_convolution_or_linear(layer[idx]):
                    # Generate file name
                    layer_name  = layer[idx]._get_name() + "_" + str(layer_names[layer[idx]._get_name()])
                    layer_names[layer[idx]._get_name()] += 1
                    output_name = source_dir + model_name + "_" + layer_name

                    # Process layer
                    lip_spectral, lip = layer_processing(lip_spectral, lip, layer[idx], output_name, relevant_layer_cnt, conv_lin_idx)
                    conv_lin_idx += 1

    print('Lipschitz spectral: {}'.format(lip_spectral))
    print('Lipschitz approximation: {}'.format(lip))

    # Store lipschitz and spectral to file
    with open(dest_dir + model_name + ".txt", "w") as fd:
        fd.write("Spectral: ")
        fd.write(str(lip_spectral.item()))
        fd.write("\nLipschitz: ")
        if isinstance(lip, torch.Tensor):
            lip = lip.item()
        fd.write(str(lip))

    return lip_spectral, lip



def layer_processing(lip_spectral, lip, layer, output_name, relevant_layer_cnt, conv_lin_idx):
    # Load from file
    print('\tDealing with {}'.format(layer._get_name()))
    use_cuda = torch.cuda.is_available()
    device = "cpu"
    U = torch.load(output_name + "_left_singular", map_location=device)
    U = adjust_for_nan(U)
    U = torch.cat(U[:n_sv], dim=0).view(n_sv, -1)
    su = torch.load(output_name + "_spectral", map_location=device)
    su = su[:n_sv]

    V = torch.load(output_name + "_right_singular", map_location=device)
    V = torch.cat(V[:n_sv], dim=0).view(n_sv, -1)
    sv = torch.load(output_name + "_spectral", map_location=device)
    sv = sv[:n_sv]

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
    print('\t    Expected: {}'.format(expected))

    lip_spectral *= expected

    # Calculate approximation
    curr, _ = optim_nn_pca_greedy(U.t() @ sigmau, sigmav @ V, use_cuda=use_cuda, max_iteration=OPTIM_ITER)
    print('\t    Approximation: {}'.format(curr))
    lip *= float(curr)
    return lip_spectral, lip


def adjust_for_nan(U):
    # Remove all NaN vectors
    out = []
    for d in U:
        if not d.isnan().any().item():
            out.append(d)
        else:
            out.append(torch.zeros_like(U[0]))
    return out
