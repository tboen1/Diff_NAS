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
    
    print('running job 200 epochs, using ADAM, lamb 1e-3')

    lamb = 1e-3
    f_name = 'baseline'
    
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
        
    d_prime = X_train_scatter.shape[0]
    
    full_data = {}

    for exp in range(2,11):
        r = int(2**exp)

        print(r)

        net = Net(1,r,c=10,d=d_prime)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        data = train_model(net, loss_fn, optimizer, lamb, 
                   train_loader, test_loader, 
                   X_train_scatter, Y_train, X_test_scatter, Y_test,
                   device, S, num_epochs=200)

        full_data[r] = data

    pk.dump(full_data, open('results/{}_data.pk'.format(f_name), 'wb'))
    #torch.save(model_data, open('results/{}_model.pth'.format(f_name), 'wb'))
        
    






    
   
