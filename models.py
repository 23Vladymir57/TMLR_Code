import gzip
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import dataloader as dl
import random
import pickle
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


class SRN(nn.Module):
    def __init__(self, hidden_dim,input_dim,output_dim,num_layers=1, activation= 'sig'):
        super(SRN, self).__init__()
        self.hidden = hidden_dim
        self.input  = input_dim
        self.output = output_dim
        self.layers = num_layers
        self.init_h = torch.eye(self.hidden)[:,0]
        self.h      = self.init_h

        if activation == 'sig':
            self.activ = torch.nn.Sigmoid()
        else:
            self.activ = torch.nn.Tanh()
        self.U      = nn.Linear(self.input,  self.hidden)
        self.W      = nn.Linear(self.hidden, self.hidden)
        self.V      = nn.Linear(self.hidden, self.output)

    def reset_h(self):
        self.h = self.init_h


    def forward(self,x):
        h = self.h
        y1 = self.U(x) #bqtch x hidden
        y2 = self.W(h)
        h = self.activ(y1+y2) # bTCH X HIDDEN
        outs = self.activ(self.V(h)) # OUTPUT LOGITS
        self.h = h
        return outs
    
class Quasy_H():
    def __init__(self):
        super(Quasy_H, self).__init__()
        self.curent   = 0
        self.previous = 0
        self.lr       = 1
    
    def add_curent(self,elem):   ### cette fonction permetra de construire une direction moyenne 
        self.curent += elem
    
    def add_previous(self,elem): ### je doute que cette fonction sera utilisée
        self.previous += elem
    
    def set_lr(self,elem):
        self.lr = elem

    def set_curent(self,elem):
        self.curent = elem
    
    def set_previous(self,elem):
        self.previous = elem
    
    def get_stats(self):
        return (self.previous, self.curent, self.lr)
        
class STATS():
    def __init__(self,lis_stats):
        super(STATS,self).__init__()
        ### this class is designed to store any kind of statistics typed as lists. It takes as input a list of strings
        self.names = lis_stats
        self.dict  = {} 
        for elem in lis_stats:
            self.dict[elem] = []
    
        
    def push(self,name,element):
        self.dict[name].append(element)

    def get_data(self):
        ls = []
        for elem in self.names:
            ls.append(self.dict[elem])
        return ls
    
class customSRN(nn.Module):
    def __init__(self, hidden_dim,input_dim,output_dim,seq_length,num_layers=1,bidirectional=False, activation= 'tanh',device='cpu',dtyp=32):
        super(customSRN, self).__init__()
        self.seqlen = seq_length
        self.hidden = hidden_dim
        self.input  = input_dim
        self.output = output_dim
        self.layers = num_layers
        self.bidire = bidirectional
        self.device = device
        self.dtyp   = dtyp
        if activation == 'tanh':
            self.activ = nn.Sigmoid()
        else:
            self.activ = nn.ReLU()
        self.U      = nn.Linear(self.input,  self.hidden).to(dtype=torch.float32, device = device)
        self.W      = nn.Linear(self.hidden, self.hidden).to(dtype=torch.float32, device = device)
        self.fc     = nn.Linear(self.hidden, self.output).to(dtype=torch.float32, device = device)

    def forward(self,x):
        self.Fbeta = {'value':[],'position':[]}
        pad_len = x[:,-1,:]
        if self.dtyp == 32:
            dtyp = torch.float32 
        else:
            dtyp = torch.float64
        x = x[:,:-1,:].to(dtype = dtyp,device=self.device)  
        h = torch.zeros(x.size(0), self.hidden).to(dtype=dtyp,device=self.device) # bqatch x hiddendi
        outs = torch.zeros(x.size(0),self.output,self.seqlen-1).to(dtype = dtyp,device=self.device) 
        mask = torch.zeros(x.size(0),self.seqlen-1).to(dtype = dtyp,device=self.device)
        x = torch.transpose(x,2,1)
        
        for j in range(x.shape[2]-1): 
            mask[:,j]= (pad_len[:,0]==j*torch.ones((x.size(0),)).to(dtype = dtyp,device=self.device)) ## cration of the padding mask
            y1 = self.U(x[:,:,j]) #bqtch x hidden
            y2 = self.W(h)
            h = self.activ(y1+y2) # bTCH X HIDDEN
            out = self.fc(h) # OUTPUT LOGITS
            outs[:,:,j] = self.activ(out)
        for k in range(outs.shape[1]):
            outs[:,k,:][mask==0] = 0        ## aplying the padding mask
        # print(outs)
        outs = torch.sum(outs,-1) ## extracting the usfull classification 
        # print(outs)
        return outs.to(dtype=torch.float64,device=self.device)
    
