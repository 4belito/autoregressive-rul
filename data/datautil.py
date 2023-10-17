import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from scipy.stats import mode
import matplotlib.pyplot as plt

import sys
sys.path.append('./data')



def sys_separation(X,sys_array):
    sys_set=set(sys_array)
    num_sys=len(sys_set)
    max_life=mode(sys_array,keepdims=True).count[0] 
    X_sep=np.full((num_sys, max_life,*X.shape[1:]), np.nan,dtype=X.dtype)
    for i,sys in enumerate(sys_set):  
        X_sys=X[sys_array==sys]
        X_sep[i,-X_sys.shape[0]:]=X_sys
    return X_sep


def fault2RUL0(df,system='System',fault='comp_f1'):
    RUL_df=pd.DataFrame()
    sys_set=set(df[system].values)
    for sys in sys_set:
        sys_df=df.loc[df[system] == sys]
        life=len(sys_df.loc[sys_df[fault]==0,fault].index)
        if sys_df.loc[sys_df[fault]==1,fault].count()>0: 
            sys_df.loc[:life,fault]=life-sys_df[:life+1].index
            sys_df.loc[life:,fault]=0
        RUL_df=pd.concat([RUL_df, sys_df], axis=0)
    return RUL_df


# # Prepare for training
# Cancel out systems that did not fail, and the samples after the fault 

def RULData_reduction(df,RUL='RUL'):
    df=df[df[RUL]!=df[RUL].shift()]
    if df.iloc[0,df.columns==RUL].item()==0.0:
        df=df.iloc[1:]
    return df


class RULDataset(Dataset):
    def __init__(self,data,sys_array,transform=None):
        x=data[0]
        y=data[1]
        self.x=x
        self.y=y
        self.sys_array=sys_array
        self.n_samples=len(x)
        self.sys_set=set(sys_array)
        self.n_sys=len(self.sys_set)
        self.n_features=x.shape[1]
        self.window=None
        self.stats=None
        self.scored=False
        self.x_origin=self.x
        self.y_origin=self.y
        self.max_life=mode(sys_array,keepdims=True).count[0]
        self.transform=transform
    
    def __getitem__(self,index):
        sample=self.x[index], self.y[index]
        if self.transform:
            sample=self.transform(sample[0]) 
        return sample
    
    def __len__(self):
        return self.n_samples
    
    def undo_x(self):
        self.x=self.x_origin
        self.window=None
        self.stats=None
    
    def undo_y(self):
        self.y=self.y_origin
        self.scored=False
        
    def temporize_pad(self,window=10):
        if not self.window:            
            dataset_temp=np.zeros([self.n_samples,window,self.n_features],dtype=self.x.dtype)
            for ind in range(self.n_samples):
                for look in range(0,window):
                    if self.sys_array[ind-look]==self.sys_array[ind]:
                        feat=self.x[ind-look,:]
                    dataset_temp[ind,window-look-1,:]=feat
            self.window=window
            self.x=dataset_temp
        else:
            print('The dataset is already temporalized. Use method undo_x first') #This is for simplicity
    
    def temporize(self,window=10):
        if not self.window: 
            x_list=[]
            y_list=[]
            new_array=[]
            w=window-1          
            for sys in self.sys_set:
                sys_ind=np.isin(self.sys_array,sys) 
                sys_life=sum(sys_ind)
                if window<=sys_life:
                    init=np.where(sys_ind)[0][w]
                    new_life=sys_life-w
                    new_array+=new_life*[sys]
                    for j in range(new_life):
                        y_list.append(self.y[init+j])
                        x_list.append(self.x[init+j-w:init+j+1,:])
                else:
                    print('window larger than life')
                    return None
            self.window=window
            self.x=np.stack(x_list,axis=0)
            self.y=np.stack(y_list,axis=0)
            self.n_samples=self.x.shape[0]
            self.sys_array=np.array(new_array)
        else:
            print('The dataset is already temporized. Use method undo_x first') #This is for simplicity

    def normalize(self,mu=0,sigma=1):
        if not self.window and not self.stats:
            self.x=(self.x-mu)/sigma
            self.stats=(mu,sigma)
        elif self.window:
            print('Normalization have to be done before temporize') #This is for simplicity
        elif self.stats:
            print('It is already normalized')
    
    def add_scores(self):
        if self.scored:
            print('Dataset already scored')
        else:
            mu_y=np.mean(self.y,axis=0)
            std_y=np.std(self.y,axis=0)    
            std_score=(self.y-mu_y)/std_y      
            self.y=np.concatenate((self.y[:,np.newaxis],std_score[:,np.newaxis]), axis=1) 
            self.scored=True
    
    def sys_separation(self):
        X_sep=sys_separation(self.x,self.sys_array)
        Y_sep=sys_separation(self.y,self.sys_array)
        sys_sep=sys_separation(self.sys_array,self.sys_array)
        return X_sep,Y_sep,sys_sep

    def cut_life(self,ini_life=0,end_life=-1):
        X_sep,Y_sep,sys_sep=self.sys_separation()        
        X_=X_sep[:,ini_life:end_life,...]
        Y_=Y_sep[:,ini_life:end_life,...]
        sys_=sys_sep[:,ini_life:end_life]
        X_=X_.reshape(-1,*X_.shape[2:])
        Y_=Y_.reshape(-1,*Y_.shape[2:])
        sys_=sys_.reshape(-1)
        not_nan=~np.isnan(X_)[:,*((len(X_.shape)-1)*[0])]
        self.x=X_[not_nan,...]
        self.y=Y_[not_nan,...]
        self.sys_array=sys_[not_nan]
        self.max_life=mode(self.sys_array,keepdims=True).count[0]
        self.x_origin=self.x
        self.y_origin=self.y


    def sub_collect(self,sys_subset):
        if set(sys_subset).issubset(self.sys_set):
            mask=np.isin(self.sys_array, list(sys_subset))
            sub_dataset=RULDataset(data=(self.x[mask],self.y[mask]),sys_array=self.sys_array[mask])
            sub_dataset.window=self.window
            sub_dataset.stats=self.stats
            sub_dataset.scored=self.scored
            sub_dataset.x_origin=self.x_origin[mask]
            sub_dataset.y_origin=self.y_origin[mask]
        else:
            print(f'Choose a sub-selection of the systems of the dataset: {self.sys_set}')
            sub_dataset=None
        return sub_dataset


# ## Data Loader Creation
# scaling_label is a lambda function
def create_loader(dataset,batch_size,window=False,shuffle=True,scored=False,stats=False):
    #Normalize
    if stats:
        mu=stats[0]
        std=stats[1]
    else:
        mu=np.mean(dataset.x,axis=0)
        std=np.std(dataset.x,axis=0)    

    dataset.normalize(mu=mu,sigma=std)   

    #temporize
    if window > 0:
        dataset.temporize(window=window)
    if scored:
        dataset.add_scores()
    
    train_loader=DataLoader(dataset=dataset, batch_size=batch_size,shuffle=shuffle)
    
    return train_loader

def create_loaders(train_dataset,test_dataset,batch_size,window=False,scored=False,stats=False):
    #Normalize
    if stats:
        mu=stats[0]
        std=stats[1]
    else:
        mu=np.mean(train_dataset.x,axis=0)
        std=np.std(train_dataset.x,axis=0)    

    train_dataset.normalize(mu=mu,sigma=std)  
    test_dataset.normalize(mu=mu,sigma=std)  

    #temporize
    if window > 0:
        train_dataset.temporize(window=window)
        test_dataset.temporize(window=window)
    if scored:
        train_dataset.add_scores()
    
    train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size[0],shuffle=True)
    test_loader=DataLoader(dataset= test_dataset, batch_size=batch_size[1],shuffle=False)
    
    return train_loader, test_loader

def create_cross_loaders(dataset,N_folds,batch_size,window=False,scored=False,shuffle=True,stats=False):
    fold_len=dataset.n_sys//N_folds+1
    sys_list=list(dataset.sys_set)
    
    #Cross validation Shufle
    if shuffle:
        random.shuffle(sys_list)
        
    train_loaders=[]
    test_loaders=[]
    for fold, fold_cut in enumerate(range(0, dataset.n_sys, fold_len)):
        test_list=sys_list[fold_cut:fold_cut+fold_len]
        train_list=list(set(sys_list)-set(test_list))
        train_dataset=dataset.sub_collect(train_list)
        test_dataset=dataset.sub_collect(test_list)        
        
        if stats:
            stats_fold=(stats[0][fold],stats[1][fold])
        else:
            stats_fold=stats
        train_loader, test_loader=create_loaders(train_dataset,test_dataset,
                                                batch_size=(batch_size,batch_size),
                                                window=window,
                                                scored=scored,
                                                stats=stats_fold)
        
        # Build DataLoader
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    return train_loaders,test_loaders

def plot_features(features):
    t=range(features.shape[1])
    for fea in range(features.shape[2]): 
        plt.title(f'Feature {fea}') 
        plt.xlabel("time") 
        plt.ylabel('value') 
        plt.plot(t,features[:,:,fea].T,color='r')
        plt.show()



