#!/usr/bin/env python
# coding: utf-8

# ## Training

# In[16]:


import torch
import torch.nn as nn
import numpy as np

import metrics as met
#import architectures as arc


# In[17]:


# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader#, random_split
# import json
# import pandas as pd

# import os

# import sys


# In[18]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ## Loss function 

# In[7]:


class RUL_Loss(nn.Module): 
    def __init__(self,loss,a=5,b=9,war=100,alpha=7,weighted=False):
        super().__init__() #RUL_Loss, self
        self.a=a
        self.b=b
        self.war=war
        self.loss=loss
        self.alpha=alpha
        self.weighted=weighted
        
    def forward(self, y_p, y_t):
        y_p = y_p
        y_t = y_t.to(y_p.dtype)  # 
        
        if self.loss=='MSE':            
            se=met.SE(war=self.war,alpha=self.alpha)
            if self.weighted:                
                res=se.weighted(y_p,y_t)
            else:
                res=se(y_p,y_t)
        
        if self.loss=='LPBP':
            lpbp=met.LPBP(a=self.a,b=self.b,war=self.war,alpha=self.alpha)
            if self.weighted:
                res=lpbp.weighted(y_p,y_t)
            else:
                res=lpbp(y_p,y_t)
        
        if self.loss=='Score':
            se=met.SE(war=self.war)
            lpbp=met.LPBP(a=self.a,b=self.b,war=self.war,alpha=self.alpha)
            if self.weighted:
                res=(se.weighted(y_p,y_t)+lpbp.weighted(y_p,y_t))/2
            else:
                res=(se(y_p,y_t)+lpbp(y_p,y_t))/2
        
        if self.loss=='diffScoring':
            diffscoring=met.DiffScoring(a=self.a,b=self.b,war=self.war,alpha=self.alpha)
            if self.weighted:
                res=diffscoring.weighted(y_p,y_t)
            else:
                res=diffscoring(y_p,y_t)
                
        if self.loss=='diff2Scoring':
            diff2scoring=met.Diff2Scoring(a=self.a,b=self.b,war=self.war,alpha=self.alpha)
            if self.weighted:
                res=diff2scoring.weighted(y_p,y_t)
            else:
                res=diff2scoring(y_p,y_t)
                
        if self.loss=='spherical':
            spherical=met.Spherical()
            res=spherical(y_p,y_t)               

        return torch.sum(res)                         


# ## Training Loop
# 
# 1. **model:** model to be trainned 
# 2. **config:** configuration of the trainning
# 3. **train_loader:** Dataloaders containning the data to learn on
# 

# In[8]:


def run_training(model,train_loader,config,noise_coef=False):
    #loss Funtion
    loss=config['Floss']         
       
    #objective
    objective = RUL_Loss(config['Floss'],a=config['a'],b=config['b'],war=config['war'],alpha=config['alpha'],weighted=config['weighted'])
    
    #optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    #trainning loop
    for m in range(config['n_epoch']):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            if noise_coef:
                y_score=y[:,1]
                noise_strench=noise_coef[0]*(y_score>0)+noise_coef[2]*(y_score<0)
                noise_bound=noise_coef[1]*(y_score>0)+noise_coef[3]*(y_score<0)
                noise=noise_strench*torch.clip(torch.abs(y_score)-noise_bound,0)*torch.randn(y.shape[0]).to(device)                         
                y=y[:,0]+torch.sign(y_score)*torch.abs(noise)  
            
            optim.zero_grad()     
            yhat = model(x)
 
            loss = objective(yhat, y.float())
            loss.backward()

            optim.step()
            
    return model #


# ## Evaluation loop

# In[9]:


def evaluar(model,test_loader):           
    #plot_test logs
    y_pred_test=np.empty(0)
    y_true_test=np.empty(0)
    
    model.eval()
    with torch.no_grad():
        #for i, (x, y) in enumerate(tqdm(test_loader[fold])):
        for (x, y) in test_loader:
            x=x.to(device)
            y=y.to(device)                 

            #prediction                
            yhat = model(x) 
            
            #plot logs
            y_pred_test=np.append(y_pred_test,yhat.to('cpu').detach().numpy())            
            y_true_test=np.append(y_true_test,y.to('cpu').detach().numpy())

    model.train()
    return y_pred_test,y_true_test




