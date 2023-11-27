#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode


# # Separate in syst lifes

def sys_separation(y,sys_array):
    sys_set=set(sys_array)
    num_syss=len(sys_set)
    max_life=mode(sys_array,keepdims=True).count[0] 
    y_sep=np.full((num_syss, max_life), np.nan)
    for i,sys in enumerate(sys_set):  
        y_sys=y[sys_array==sys]
        y_sep[i,-y_sys.shape[0]:]=y_sys
    return y_sep

def sys_separation_dec(y,sys_array):
    sys_set=set(sys_array)
    num_syss=len(sys_set)
    max_life=mode(sys_array,keepdims=True).count[0] 
    y_sep=np.full((num_syss, max_life,y.shape[1]), np.nan)
    for i,sys in enumerate(sys_set):  
        y_sys=y[sys_array==sys]
        y_sep[i,-y_sys.shape[0]:,:]=y_sys
    return y_sep


def scoring(x,y,a=10,b=13):
    s=x-y
    return np.exp(s / a)*(s > 0) +np.exp(-s /b)*(s <= 0)-1


def plot_fig(x_eval,
            start=0,
            title='title',
            xlabel='xlabel',
            ylabel='ylabel',
            plots={'true':np.nan,'mean':np.nan, 'lower_std':np.nan,'upper_std':np.nan}):
    fig, axis = plt.subplots(figsize=(28, 15))
    cross_life=plots[next(iter(plots))].shape[0] 
    true=range(cross_life-1,-1,-1)[-start:]
    plt.fill_between(x_eval, plots['lower_std'][-start:] , plots['upper_std'][-start:] , alpha=0.3, color='b', label='Confidence Interval')
    
    for label,curve in plots.items(): 
        if label=='true':
            true=range(cross_life-1,-1,-1)[-start:]
            axis.plot(x_eval,curve[-start:],linestyle='dashed',color='black',label=label)   
        elif label not in ['true','lower_std','upper_std']:
            axis.plot(x_eval,curve[-start:],color='#ff7f00',label=label)   
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.invert_xaxis()
    #plt.savefig(f'{address}/{title}.png')
    plt.show() 



# # Plot Figure
def plot_error_fig(start=0,
            title='title',
            xlabel='xlabel',
            ylabel='ylabel',
            plots={'true':np.nan,'mean':np.nan, 'lower_std':np.nan,'upper_std':np.nan}):
    fig, axis = plt.subplots(figsize=(28, 15))
    cross_life=plots[next(iter(plots))].shape[0] 
    true=range(cross_life-1,-1,-1)[-start:]
    plt.fill_between(true, plots['lower_std'][-start:] , plots['upper_std'][-start:] , alpha=0.3, color='b', label='Confidence Interval')
    
    for label,curve in plots.items(): 
        if label=='true':
            true=range(cross_life-1,-1,-1)[-start:]
            axis.plot(true,curve[-start:],linestyle='dashed',color='black',label=label)   
        elif label not in ['true','lower_std','upper_std']:
            axis.plot(true,curve[-start:],color='#ff7f00',label=label)   
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.invert_xaxis()
    #plt.savefig(f'{address}/{title}.png')
    plt.show() 



# # Error Plot

def error_plot(y_pred,y_true):
    error=y_pred-y_true      
    error_under=error.copy()
    error_over=error.copy()
    error_under[error>0]=0
    error_over[error<0]=0
    
    error_under=error.copy()
    error_over=error.copy()
    error_under[error>0]=0
    error_over[error<0]=0    
    return error_under,error_over





