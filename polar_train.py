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
    
    print('fixing seed = 0 for reproducibility')
    print('running sanity check 5 epochs, using ADAM, lamb 1')
    
    torch.manual_seed(0)

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
    
    #sketch_compute, R = pre_compute_sketch(X_train_scatter, max_size=1000)
    
    d_prime = X_train_scatter.shape[0]
    
    full_data = {}
    model_data = {}

    start_neuron = 100
    max_neuron = 105
    
    for r in range(start_neuron, max_neuron):
        
        if r == start_neuron:
            net = Net(3,r,c=10,d=d_prime).to(device)
        else:
            w, polar, z = get_w(precomputed, X_train_scatter, Y_train, net, lamb, c = 10, exact = True, R = None)
            #w, polar, _ = get_w(sketch_compute, X_train_scatter, Y_train, net, lamb, c = 10, exact = False, R = R)
            tau4 = closed_form_tau(X_train_scatter, Y_train, net, lamb, w)
            
            full_data['{}'.format(r)] = {'model': net.state_dict(), 'optimizer': optimizer.state_dict()}

            
            print('R: {}, POLAR: {:.4f}, TAU^4: {:.4f}'.format(r, polar, tau4))
            
            net = update_weights(net, tau4, w)
            net = net.to(device)
            
            W = [] 
            for i in range(len(net.layers)): W.append(net.layers[i].weight.data.numpy())
            
            full_data['{}'.format(r)] = {}
            full_data['{}'.format(r)]['training'] = data
            full_data['{}'.format(r)]['metrics'] = {'polar': polar.item(), 'tau4': tau4.detach().item()}
            full_data['{}'.format(r)]['weights'] = W
        
        lr = 1e-4
        lamb = 1
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

        loss, acc = loss_fn(net, X_train_scatter.to(device), Y_train.to(device), lamb)
        print('')
        print('-'*40)
        print(loss, acc)
        print('-'*40)
        print('')
        
        data = train_model(net, loss_fn, optimizer, lamb, 
                   train_loader, test_loader, 
                   X_train_scatter, Y_train, X_test_scatter, Y_test,
                   device, S, num_epochs = 5)

        loss, acc = loss_fn(net, X_train_scatter.to(device), Y_train.to(device), lamb)
        print('')
        print('-'*40)
        print(loss, acc)
        print('-'*40)
        print('')


    f_name = 'polar_scatter_true'.format(r)

    pk.dump(full_data, open('results/{}_data.pk'.format(f_name), 'wb'))
    #torch.save(model_data, open('results/{}_model.pth'.format(f_name), 'wb'))
        
    






    
   
