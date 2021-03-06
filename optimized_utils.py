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
    

def convert_image(image_tensor, d = 784):
    '''vectorizes batch of images into [batch, d]'''
    
    size = image_tensor.shape[0]
    return torch.transpose(image_tensor.squeeze().flatten().reshape(size, d),0,1)

def convert_label(label_tensor, c = 10):
    '''1 hot encodes label into tensor c x N'''
    
    label_tensor = torch.nn.functional.one_hot(label_tensor, num_classes=c) 
    label_tensor[label_tensor == 0] = -1
    return torch.transpose(label_tensor,0,1).type(torch.float32)
    
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
    var_0 = torch.norm(net.layers[0].weight@X, p=2, dim=1).unsqueeze(1)
    total = var_n
    for i in range(hidden):
        total = total@abs(net.layers[i+1].weight)
    total = total@var_0
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

def output(epoch, batch_index, num_batches, batch_loss, batch_acc, train_loss, train_acc, test_loss, test_acc):
    str = 'epoch: {}, {}/{}, | BATCH loss: {:.3f}, acc {:.3f} | TRAIN loss: {:.3f}, acc {:.3f} | TEST loss: {:.3f}, acc {:.3f}'
    print(str.format(epoch, batch_index, num_batches,
                        batch_loss, batch_acc, 
                        train_loss, train_acc, 
                        test_loss, test_acc)) 

def get_scattering_transform(J=2,L=8, device='cpu'):
    '''current implementation of J/L maps d = 28 x 28 --> d_prime = 3969'''

    S = Scattering2D(J=J, shape = (28,28), L=L, max_order=2)
    
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
    
def train_model(net, loss_fn, optimizer, lamb, train_loader, test_loader, 
                        X_train_full, Y_train_full, X_test_full, Y_test_full, 
                        device, S):

    train_step = make_train_step(net, loss_fn, optimizer, lamb)
    
    batch_loss_full = []
    batch_acc_full = []
    train_loss_full = []
    train_acc_full = []
    test_loss_full = []
    test_acc_full = []
    
    for epoch in range(20):
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
    
                  Y = Y_test_full.to(device)
                  X = X_test_full.to(device) 
    
                  test_loss, test_acc = train_step(X, Y, 'eval')  
                  test_loss_full.append(test_loss)
                  test_acc_full.append(test_acc)  
            
              if  batch_idx % 25 == 0:
                  output(epoch, batch_idx, len(train_loader), batch_loss, batch_acc, 
                         train_loss, train_acc, test_loss, test_acc) 
    
    data = {}
    data['batch_loss'] = np.asarray(batch_loss_full)
    data['batch_acc'] = np.asarray(batch_acc_full)
    data['train_loss'] = np.asarray(train_loss_full)
    data['train_acc'] = np.asarray(train_acc_full)
    data['test_loss'] = np.asarray(test_loss_full)
    data['test_acc'] = np.asarray(test_acc_full)
                         
    return data 
    
####################################  Polar update code  ####################################  
def pre_compute(X):
    return torch.pinverse(X@torch.transpose(X,0,1))@X

def get_w(precomputed, trainloader, net, lamb, c = 10):
    
    #optimized version of calculating polar
    norm = torch.zeros((c,2))
    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
        Y = convert_label(Y_batch).to(device)
        X = scatter_transform_tensor(S, X_batch, device = device).to(device)
    
        Q_m = (1/lamb)*(Y - new_net(X))
        z[:,batch_idx*32:(batch_idx+1)*32] = Q_m
    
        norm[:,0] += torch.norm(F.relu(-Q_m), dim = 1, p = 'fro')**2
        norm[:,1] += torch.norm(F.relu(Q_m), dim = 1, p = 'fro')**2
        
    norm[:,0] = norm[:,0]**0.5
    norm[:,1] = norm[:,1]**0.5

    index = torch.argmax(norm, keepdim=True)
    polar = norm.flatten()[index]
    
    col = int(index%2)
    row = int((index - col)/2)
    
    #initializing z
    z = torch.zeros((1,len(trainloader.dataset)))
    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader): 
        Y = convert_label(Y_batch).to(device)
        X = scatter_transform_tensor(S, X_batch, device = device).to(device)
    
        Q_m = (1/lamb)*(Y - new_net(X))
        z[:,batch_idx*32:(batch_idx+1)*32] = Q_m[row,:]
    
    u_star = torch.zeros([c,1])
    
    if col == 1: #implies that \|Q_+\|_F > \|Q_-\|_F
        u_star[row] = 1
        z_star = (F.relu(z)/polar).unsqueeze(0)
    
    if col == 0: #implies that \|Q_-\|_F > \|Q_+\|_F
        u_star[row] = -1
        z_star = (F.relu(-z)/polar).unsqueeze(0)
    
    w1 = precomputed@torch.transpose(z_star,0,1)
    
    norm_constant = torch.zeros([1,1])
    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader): 
        X = scatter_transform_tensor(S, X_batch, device = device).to(device)
        norm_constant += torch.norm(F.relu(torch.transpose(w1,0,1)@X),2)**2 
        
    w1 = w1/(norm_constant**0.5)
    
    w = [w1]
    for i in range(net.num_hidden_layers): w.append(torch.ones([1,1]))
    w.append(u_star)
    
    return w, polar.detach().item()
    
def closed_form_tau(train_loader, net, lamb, w):

    reg_opt = torch.zeros([1,1])
    term1_opt = torch.zeros([1,1])
    term2_opt = torch.zeros([1,1])
    
    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader): 
            Y = convert_label(Y_batch).to(device)
            X = scatter_transform_tensor(S, X_batch, device = device).to(device)
    
            w_pred_m = w[-1]@F.relu(torch.transpose(w[0],0,1)@X)
            W_pred_m = net(X)
    
            reg += torch.norm(F.relu(torch.transpose(w[0],0,1)@X),2)**2
            term1_opt += w_pred_m.flatten().unsqueeze(0)@torch.transpose(Y.flatten().unsqueeze(0),0,1)
            term2_opt += w_pred_m.flatten().unsqueeze(0)@torch.transpose(W_pred_m.flatten().unsqueeze(0),0,1)


    reg_opt = lamb*(reg_opt**0.5)
    
    tau4 = term1_opt - term2_opt - reg_opt

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
    






    
   
