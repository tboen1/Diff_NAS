import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from kymatio.torch import Scattering2D

def get_dataloaders(dataset, train_size = None, test_size = None, minibatch_size = None, get_full_datasets = False):
    if dataset == 'MNIST':
        if train_size is None:
            train_size = 60000
        if test_size is None:
            test_size = 10000
        if minibatch_size is None:
            minibatch_size = 32
            
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
        data_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        data_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
    if dataset == 'CIFAR10':
        if train_size is None:
            train_size = 50000
        if test_size is None:
            test_size = 10000
        if minibatch_size is None:
            minibatch_size = 32
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        data_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        data_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    
    train_subset = torch.utils.data.Subset(data_trainset, range(0,train_size))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=minibatch_size, shuffle=True)
    
    test_subset = torch.utils.data.Subset(data_testset, range(0,test_size))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=minibatch_size, shuffle=True)
    
    if get_full_datasets:
        full_train = torch.utils.data.DataLoader(train_subset, batch_size=train_size, shuffle=True)
        full_test = torch.utils.data.DataLoader(test_subset, batch_size=test_size, shuffle=True)
        
        X_train, Y_train = iter(full_train).next()
        X_test, Y_test = iter(full_test).next()
        return train_loader, test_loader, X_train, Y_train, X_test, Y_test
    else:
        return train_loader, test_loader
    
class Net(nn.Module):
    '''generalized FNN. Tunable number of hidden layers and width'''
    
    def __init__(self, num_hidden_layers, r, c=10, d=784):
        super(Net, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.r = r
        self.c = c
        self.d = d
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d,r, bias = False))
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(r,r, bias = False))
        self.layers.append(nn.Linear(r,c, bias = False))
    
    def forward(self, x):
        out = torch.transpose(x,0,1)
        out = self.layers[0](out)
        out = F.relu(out)
        for i in range(self.num_hidden_layers):
            out = self.layers[i+1](out)
            out = F.relu(out)
        out = self.layers[self.num_hidden_layers + 1](out)

        return torch.transpose(out,0,1)
        
def loss_fn(net, X, Y, lamb):
    '''flexible loss function. Can work on a net with any number of hidden layers'''
    
    hidden = net.num_hidden_layers

    #reg term
    var_n = torch.norm(net.layers[hidden+1].weight, p=1, dim=0).unsqueeze(0)
    var_0 = torch.norm(F.relu(net.layers[0].weight@X), p=2, dim=1).unsqueeze(1)
    total = var_0
    for i in range(hidden):
        total = abs(net.layers[i+1].weight)@total
    total = var_n@total
    reg = torch.sum(total.flatten(0))

    #pred term
    pred = net(X)
    pred_loss = 0.5*torch.norm(Y - pred, 'fro')**2
    acc = get_acc(pred, Y)

    loss = pred_loss + lamb*reg
    
    return loss, acc

def get_acc(y_pred, Y):
    '''calculates accuracy'''
    
    y_pred[y_pred<=0]=-1
    y_pred[y_pred>0]=1
    
    scores = Y - y_pred
    scores_sum = torch.sum(abs(scores), dim=0)

    nonzeros = torch.norm(scores_sum,0)

    acc = 1-(nonzeros/scores_sum.shape[0])

    return acc.detach().item()
    
def make_train_step(net, loss_fn, optimizer, lamb):
    def train_step(X, Y, mode):
        if mode == 'train':
          net.train()

          loss, acc = loss_fn(net, X, Y, lamb)
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

        if mode == 'eval':
          net.eval()
          loss, acc = loss_fn(net, X, Y, lamb)
        
        return loss.detach().item(), acc
    return train_step

def output(epoch, batch_index, num_batches, batch_loss, batch_acc, train_loss, train_acc, test_loss, test_acc, polar):
    str = 'epoch: {}, {}/{}, | BATCH loss: {:.3f}, acc {:.3f} | TRAIN loss: {:.3f}, acc {:.3f} | TEST loss: {:.3f}, acc {:.3f} | polar: {:.3f}'
    print(str.format(epoch, batch_index, num_batches,
                        batch_loss, batch_acc, 
                        train_loss, train_acc, 
                        test_loss, test_acc, 
                        polar)) 
    
def train_model(net, loss_fn, optimizer, lamb, train_loader, test_loader, 
                        X_train_full, Y_train_full, X_test_full, Y_test_full, 
                        device, S, num_epochs=30):

    train_step = make_train_step(net, loss_fn, optimizer, lamb)
    
    batch_loss_full = []
    batch_acc_full = []
    train_loss_full = []
    train_acc_full = []
    test_loss_full = []
    test_acc_full = []
    polar_vals = []
    
    for epoch in range(num_epochs):
          for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
              
              Y = convert_label(Y_batch).to(device)
              X = scatter_transform_tensor(S, X_batch, device = device).to(device)
    
              batch_loss, batch_acc = train_step(X, Y, 'train')
              batch_loss_full.append(batch_loss)
              batch_acc_full.append(batch_acc)
          
              with torch.no_grad(): #calculating over entire datasets
                  
                  Y = Y_train_full.to(device)
                  X = X_train_full.to(device) 
    
                  train_loss, train_acc = train_step(X, Y, 'eval')
                  train_loss_full.append(train_loss)
                  train_acc_full.append(train_acc)
                
                  polar = get_polar(X, Y, net, lamb, verbose = False)
                  polar = polar.item()
                  polar_vals.append(polar)
    
                  Y = Y_test_full.to(device)
                  X = X_test_full.to(device) 
    
                  test_loss, test_acc = train_step(X, Y, 'eval')  
                  test_loss_full.append(test_loss)
                  test_acc_full.append(test_acc)  
                
            
              if  batch_idx % 25 == 0:
                  output(epoch, batch_idx, len(train_loader), batch_loss, batch_acc, 
                         train_loss, train_acc, test_loss, test_acc, polar) 
    
    data = {}
    data['batch_loss'] = np.asarray(batch_loss_full)
    data['batch_acc'] = np.asarray(batch_acc_full)
    data['train_loss'] = np.asarray(train_loss_full)
    data['train_acc'] = np.asarray(train_acc_full)
    data['test_loss'] = np.asarray(test_loss_full)
    data['test_acc'] = np.asarray(test_acc_full)
    data['polar'] = np.asarray(polar_vals)
                         
    return data 
    
def d_prime_simulator(J, L, shape):
    C, N_1, N_2 = shape
    d_prime = C*(1 + L*J + 0.5*J*(J-1)*L**2)*(N_1/(2**J))*(N_2/(2**J))
    return d_prime
    
def get_scattering_transform(J=2,L=8, shape = (28,28), device='cpu'):
    '''current implementation of J/L maps d = 28 x 28 --> d_prime = 3969'''

    S = Scattering2D(J=J, shape = shape, L=L, max_order=2)
    
    if device == 'cuda':
        S.cuda()
        
    return S
        
def scatter_transform_tensor(S, tensor, device='cpu'):
    '''
    tensor_dim = [batch, channels, height, width]
    transform_dim = [batch, d_prime]
    '''
    Sx = S.forward(tensor.to(device))
    scatter = torch.transpose(Sx.reshape([tensor.shape[0],Sx.shape[1]*Sx.shape[2]*Sx.shape[3]*Sx.shape[4]]),0,1)
    return scatter
    
def convert_image(image_tensor, d = 784):
    '''vectorizes batch of images into [batch, d]'''
    
    size = image_tensor.shape[0]
    return torch.transpose(image_tensor.squeeze().flatten().reshape(size, d),0,1)

def convert_label(label_tensor, c = 10):
    '''1 hot encodes label into tensor c x N'''
    
    label_tensor = torch.nn.functional.one_hot(label_tensor, num_classes=c) 
    label_tensor[label_tensor == 0] = -1
    return torch.transpose(label_tensor,0,1).type(torch.float32)
    
####################################  Polar update code  ####################################  
def pre_compute(X):
    X = X.type(torch.float64)
    return torch.pinverse(X.T)

def pre_compute_sketch(X, max_size):
    p = max_size 
    N = X.shape[1]

    X = X.type(torch.float64)
    R = torch.normal(0,1/p,(p,N)).type(torch.float64)
    B = R@X.T

    return torch.pinverse(B), R

def get_polar(X, Y, net, lamb, verbose = False):
    with torch.no_grad():
        Q = (1/lamb)*(Y - net(X)) #calculates gradient

        norm = torch.zeros((Q.shape[0],2))
        norm[:,0] = torch.norm(F.relu(-Q), dim = 1, p = 'fro')
        norm[:,1] = torch.norm(F.relu(Q), dim = 1, p = 'fro')
    
        index = torch.argmax(norm, keepdim=True)
        polar = norm.flatten()[index]

        col = int(index%2)
        row = int((index - col)/2)

    if verbose:
        return polar, row, col, Q
    else:
        return polar

def get_z_u(polar, row, col, Q):
    u_star = torch.zeros([Q.shape[0],1])
        
    if col == 1: #implies that \|Q_+\|_F > \|Q_-\|_F
        u_star[row] = 1
        z_star = (F.relu(Q[row,:])/polar).unsqueeze(0)
    
    if col == 0: #implies that \|Q_-\|_F > \|Q_+\|_F
        u_star[row] = -1
        z_star = (F.relu(-Q[row,:])/polar).unsqueeze(0)

    return z_star, u_star

def check_optimality(v_hat, polar, Q, row, X):
    z_hat = v_hat.T@X
    z_hat = z_hat/torch.norm(F.relu(z_hat),2)

    Q_i = Q[row,:]

    polar_hat = torch.norm(Q_i@z_hat.T, p='fro')

    return polar_hat, polar

def sketching_solution(z, R, sketch_compute):
    z = z.type(torch.float64)
    c = R@z.T

    v = sketch_compute@c 

    return v.type(torch.float32)

def exact_solution(z, precomputed):
    z = z.type(torch.float64)
    v = precomputed@torch.transpose(z,0,1)

    return v.type(torch.float32)

def get_v_optimal(precomputed, X, Y, net, lamb):

    polar, row, col, Q = get_polar(X, Y, net, lamb, verbose = True)
    z_star, u_star = get_z_u(polar, row, col, Q)

def get_w(precomputed, X, Y, net, lamb, c = 10, exact = True, R = None):
    with torch.no_grad():
        Q = (1/lamb)*(Y - net(X)) #calculates gradient
      
        norm = torch.zeros((c,2))
        norm[:,0] = torch.norm(F.relu(-Q), dim = 1, p = 'fro')
        norm[:,1] = torch.norm(F.relu(Q), dim = 1, p = 'fro')
    
        index = torch.argmax(norm, keepdim=True)
        polar = norm.flatten()[index]
        
        col = int(index%2)
        row = int((index - col)/2)
        
        u_star = torch.zeros([c,1])
        
        if col == 1: #implies that \|Q_+\|_F > \|Q_-\|_F
            u_star[row] = 1
            z_star = (F.relu(Q[row,:])/polar).unsqueeze(0)
        
        if col == 0: #implies that \|Q_-\|_F > \|Q_+\|_F
            u_star[row] = -1
            z_star = (F.relu(-Q[row,:])/polar).unsqueeze(0)

        if exact:
            w1 = exact_solution(z_star, precomputed)
            w1 = w1/torch.norm(F.relu(torch.transpose(w1,0,1)@X),2) 

        else:
            w1 = sketching_solution(z_star, R, precomputed)
            w1 = w1/torch.norm(F.relu(torch.transpose(w1,0,1)@X),2)
        
        polar_hat, polar = check_optimality(w1, polar, Q, row, X)

        print(polar_hat.item(), polar.item(), polar_hat.item()/polar.item())

        w = [w1]
        for i in range(net.num_hidden_layers): w.append(torch.ones([1,1]))
        w.append(u_star)
    
    return w, polar, z_star
    
def closed_form_tau(X, Y, net, lamb, w, device):

    for i in range(len(w)):
        w[i] = w[i].to(device)
    
    with torch.no_grad():
        W_pred = net(X)
        reg = lamb*torch.norm(F.relu(torch.transpose(w[0],0,1)@X),2)
        w_pred = w[-1]@F.relu(torch.transpose(w[0],0,1)@X)
        #w_pred = w[-1]@F.relu(w[3]@F.relu(w[2]@F.relu(w[1]@(F.relu(torch.transpose(w[0],0,1)@X)))))
    
        top = torch.trace(w_pred@torch.transpose(Y,0,1)) - torch.trace(w_pred@torch.transpose(W_pred,0,1)) - reg
        bottom = torch.trace(w_pred@torch.transpose(w_pred,0,1))
    
        tau4 = top/bottom

    return tau4
    
def update_weights(net, tau4, w):
    with torch.no_grad():
        h,r,c,d = net.num_hidden_layers, net.r, net.c, net.d

        #calculating scaling factors
        w1_norm = torch.norm(w[0],2)
        scale_factor = (tau4/w1_norm)**(1/(net.num_hidden_layers+1))
        w1_scale_factor = scale_factor/w1_norm

        #updating weights
        W = []
        for i in range(len(net.layers)): W.append(net.layers[i].weight.data)

        W[0] = torch.cat((W[0], w1_scale_factor*torch.transpose(w[0],0,1)),dim=0)
        W[-1] = torch.cat((W[-1], scale_factor*w[-1]),dim=1)

        for i in range(1,h+1):
            temp = torch.zeros((r+1,r+1))
            temp[:r,:r] = W[i]
            temp[r,r] = scale_factor
            W[i] = temp

    new_net = Net(h,r+1,c=c,d=d)

    for i in range(len(new_net.layers)): new_net.layers[i].weight = nn.Parameter(W[i], requires_grad = True)

    return new_net
    






    
   
