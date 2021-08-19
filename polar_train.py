import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
from kymatio.torch import Scattering2D

from importlib import reload

import naive_utils 
reload(naive_utils)
from naive_utils import *

if __name__ == '__main__':
    print('STARTING JOB')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader, test_loader, X_train, Y_train, X_test, Y_test = get_dataloaders("MNIST", train_size = 2000,
                                                                              test_size = 1000,
                                                                              minibatch_size = 32,
                                                                              get_full_datasets = True)
                                                                              
    S = get_scattering_transform(device = device)
    
    X_train_scatter = scatter_transform_tensor(S, X_train, device=device)
    X_test_scatter = scatter_transform_tensor(S, X_test, device=device)
    
    Y_train = convert_label(Y_train)
    Y_test = convert_label(Y_test)
    
    precomputed = pre_compute(X_train_scatter)    
    
    d_prime = X_train_scatter.shape[0]
    
    full_data = {}
    model_data = {}
    
    for r in range(start_neuron, max_neuron):
        
        if r == start_neuron:
            net = Net(3,r,c=10,d=d_prime)
        else:
            w, polar = get_w(precomputed, X_train_scatter, Y_train, net, lamb, c = 10)
            tau4 = closed_form_tau(X_train_scatter, Y_train, net, lamb, w)
            
            full_data['{}'.format(r)] = {'model': net.state_dict(), 'optimizer': optimizer.state_dict()}

            
            print('R: {}, POLAR: {:.4f}, TAU^4: {:.4f}'.format(r, polar, tau4))
            
            net = update_weights(net, tau4, w)
            
            full_data['{}'.format(r)] = {}
            full_data['{}'.format(r)]['training'] = data
            full_data['{}'.format(r)]['metrics'] = {'polar': polar.detach().item(), 'tau4': tau4.detach().item()}
        
        lr = 1e-3
        lamb = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        
        data = train_model(net, loss_fn, optimizer, lamb, 
                   train_loader, test_loader, 
                   X_train_scatter, Y_train, X_test_scatter, Y_test,
                   device, S)

    pk.dump(full_data, open(f_name, 'wb'))
    torch.save(model_data, open(f_name, 'wb'))
        
    






    
   
