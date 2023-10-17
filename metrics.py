#!/usr/bin/env python
# coding: utf-8

# ## Performance metrics

# In[2]:


import torch
import torch.nn as nn


# ## Sequared Error (SE)

# In[3]:


class SE():
    
    def __init__(self,war=500,alpha=7):       
        self.war=war        
        self.alpha=alpha
    def __call__(self,x,y):
        return (x-y)**2
       
    def weighted(self,x,y):
        sofmax=nn.Softmax(dim=0)
        return self(x,y)*torch.exp(self.alpha*(1-y/self.war))
        


# ## Late prediction based penalization (LPBP) 

# In[4]:


class LPBP():
    #b>a
    def __init__(self,a=10,b=13,war=500,alpha=7):
        self.a=a
        self.b=b        
        self.war=war
        self.alpha=alpha
        
    def __call__(self,x,y):
        s=x-y
        return torch.exp(s / self.a)*(s > 0).to(torch.float32) +torch.exp(-s / self.b)*(s <= 0).to(torch.float32)-1
       
    def weighted(self,x,y):
        #sofmax=nn.Softmax(dim=0)
        return self(x,y)*torch.exp(self.alpha*(1-y/self.war))
        


# ## Differentiable Scoring

# In[5]:


class DiffScoring():
    #b>a
    def __init__(self,a=10,b=13,war=500,alpha=7):
        self.a=a
        self.b=b        
        self.war=war
        self.alpha=alpha
        
    def __call__(self,x,y):
        s=x-y
        return (torch.exp(s / self.a)-s/self.a)*(s > 0).to(torch.float32) +(torch.exp(-s / self.b)+s/self.b)*(s <= 0).to(torch.float32)-1
       
    def weighted(self,x,y):
        #sofmax=nn.Softmax(dim=0)
        return self(x,y)*torch.exp(self.alpha*(1-y/self.war))


# ## Differentiable2 Scoring

# In[6]:


class Diff2Scoring():
    #b>a
    def __init__(self,a=10,b=13,war=500,alpha=7):
        self.a=a
        self.b=b        
        self.war=war
        self.alpha=alpha
        
    def __call__(self,x,y):
        s=x-y
        return (torch.exp(s / self.a)-s/self.a-s**2/self.a**2/2)*(s > 0).to(torch.float32) +(torch.exp(-s / self.b)+s/self.b-s**2/self.b**2/2)*(s <= 0).to(torch.float32)-1
       
    def weighted(self,x,y):
        #sofmax=nn.Softmax(dim=0)
        return self(x,y)*torch.exp(self.alpha*(1-y/self.war))


# ## Spherical metric

# In[34]:


class Spherical():        
    def __call__(self,x,y):
        return (x-y)**2/((1+x**2)*(1+y**2))

    def weighted(self,x,y):
        return self(x,y)


# ## Nasa score

# In[35]:


def Nasa_score(x,y,a=10,b=13):
    lpbp=LPBP(a,b)
    se=SE()
    s=torch.sum(lpbp(x,y))
    rmse=torch.sqrt(torch.mean(se(x,y)))
    return (s+rmse)/2





