
import math
import torch
import torchvision

import numpy as np

from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method

from seqlip import optim_nn_pca_greedy

from experiments.model_get_sv import compute_module_input_sizes, execute_through_model, save_singular, spec_mnist


n_sv = 200

def get_lipschitz(model, out_dir, model_name):   
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
    return model_operations(model)





def model_operations(model):
    """ Starting point of the script is a saving of all singular values and vectors
    in mnist_save/

    We perform the 100-optimization implemented in optim_nn_pca_greedy
    """
    for p in model.parameters():
        p.requires_grad = False

    compute_module_input_sizes(model, [1, 1, 28, 28])

    lip = 1
    lip_spectral = 1


    # Indices of convolutions and linear layers
    convs = [1, 2, 3, 4]

    for i in range(len(convs) - 1):
        print('Dealing with convolution {}'.format(i))
        U = torch.load('mnist_save/conv{}-left-singular'.format(convs[i]))
        U = torch.cat(U[:n_sv], dim=0).view(n_sv, -1)
        su = torch.load('mnist_save/conv{}-singular'.format(convs[i]))
        su = su[:n_sv]

        V = torch.load('mnist_save/conv{}-right-singular'.format(convs[i+1]))
        V = torch.cat(V[:n_sv], dim=0).view(n_sv, -1)
        sv = torch.load('mnist_save/conv{}-singular'.format(convs[i+1]))
        sv = sv[:n_sv]
        print('Ratio layer i  : {:.4f}'.format(float(su[0] / su[-1])))
        print('Ratio layer i+1: {:.4f}'.format(float(sv[0] / sv[-1])))

        U, V = U.cpu(), V.cpu()


        if i == 0:
            sigmau = torch.diag(torch.Tensor(su))
        else:
            sigmau = torch.diag(torch.sqrt(torch.Tensor(su)))

        if i == len(convs) - 2:
            sigmav = torch.diag(torch.Tensor(sv))
        else:
            sigmav = torch.diag(torch.sqrt(torch.Tensor(sv)))

        expected = sigmau[0,0] * sigmav[0,0]
        print('Expected: {}'.format(expected))

        lip_spectral *= expected

        curr, _ = optim_nn_pca_greedy(sigmav @ V, U.t() @ sigmau)
        print('Approximation: {}'.format(curr))
        lip *= float(curr)


    print('Lipschitz spectral: {}'.format(lip_spectral))
    print('Lipschitz approximation: {}'.format(lip))
    return lip_spectral, lip


