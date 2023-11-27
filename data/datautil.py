import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from scipy.stats import mode
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
sys.path.append('./data')



def sys_separation(X,sys_array,aligned='left'):
    sys_set=set(sys_array)
    num_sys=len(sys_set)
    max_life=mode(sys_array,keepdims=True).count[0] 
    X_sep=np.full((num_sys, max_life,*X.shape[1:]), np.nan,dtype=X.dtype)
    
    if aligned=='left':
        slicer=lambda x: slice(None,x)
    elif aligned=='right':
        slicer=lambda x: slice(-x,None)
    else:
        print('aligned argument needs to be equalt left or right')
    
    for i,sys in enumerate(sys_set):  
        X_sys=X[sys_array==sys]
        sys_life=X_sys.shape[0]
        X_sep[i,slicer(sys_life)]=X_sys
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

def moving_average(data,window_size,dim):
    moving_avg = []
    for i in range(data.shape[dim] - window_size + 1):
        window = np.take(data, indices=list(range(i, i + window_size)), axis=dim)
        avg = np.mean(window,axis=dim,keepdims=True)
        moving_avg.append(avg)
    return np.concatenate(moving_avg,axis=dim)

def sequential_data(data_sep):
    n_sys=data_sep.shape[0]
    longest_life=data_sep.shape[1]
    X_recover=data_sep.reshape(n_sys*longest_life,-1)
    
    # Find rows with NaN values
    no_nan_rows = ~np.isnan(X_recover).any(axis=1)
    sys_array=np.array([i for i in range(n_sys) for _ in range(longest_life)])

    return X_recover[no_nan_rows],sys_array[no_nan_rows]

def compute_stats(var, norm_type='minmax'):
    if len(var):
        axs=tuple(range(var.ndim - 1))
        if norm_type == 'minmax':
            stats=np.min(var,axis=axs),np.max(var,axis=axs)
        elif norm_type == 'normal':
            stats=np.mean(var,axis=axs),np.std(var,axis=axs)
        else:
            print('Argument type needs to be minmax or normal')
    else:
        stats=None,None
    return stats

def normalize(var,norm_type,stats):
    if len(var):
        if norm_type == 'normal':
            mu,sigma=stats
            var=(var-mu)/sigma
        elif norm_type == 'minmax':
            min,max=stats
            var=(var-min)/(max-min)
        else:
            print('Argument norm_type needs to be minmax or normal')
    else:
        print('no variable was found')
    return var

def RULData_reduction(df,RUL='RUL'):
    df=df[df[RUL]!=df[RUL].shift()]
    if df.iloc[0,df.columns==RUL].item()==0.0:
        df=df.iloc[1:]
    return df


class RULDataset_fast(Dataset):
    def __init__(self,data,sys_array,data_labels,data_name=None,dataset_name=None,transform=None):
        """_summary_

        Args:
            data (np.arrays(N,n_features),np.arrays(N,n_latents),np.arrays(n_syst,sim_len,n_inputs),np.arrays(N,n_targets)) :  having the features,latents,inputs,targets of the dataset 
            sys_array (np.array): sequence of system numbers of each measurement i.e 0,0,0,1,1,1,2,2,2 
            transform (_type_, optional): _description_. Defaults to None.
            variable_name (_type_, optional): _description_. Defaults to None.
            target_name (_type_, optional): _description_. Defaults to None.
            input_name (_type_, optional): _description_. Defaults to None.
            latent_name (_type_, optional): _description_. Defaults to None.
            data_name (_type_, optional): _description_. Defaults to None.
            dataset_name (_type_, optional): _description_. Defaults to None.
        """        

        #Variables
        var_key=list(data_labels.keys())
        self.var=dict(zip(var_key, data)) 
        self.sys_array=sys_array

        #Numerical information
        self.n_var={var_key[i]:data[i].shape[-1] for i in range(4)}        
        self.sys_list=list(set(self.sys_array))
        self.sys_list.sort()
        self.n_samples=len(self.var['X']) #improve this
        self.n_sys=len(self.sys_list)
        #self.max_life=mode(self.sys_array,keepdims=True).count[0]
        
        # Structure data
        self.window={}
        self.type_stats=[]
        #self.scored=False 

        #For recovering with reset method
        self.origin=self.var.copy()
        self.sys_array_origin=self.sys_array.copy()
        
        #Information
        self.data_name=data_name
        self.dataset_name=dataset_name
        self.labels=data_labels
        
        # Prepare for sampling
        self.transform=transform
        if self.n_var['U']: self.var['U']=self.prepare_u()

    
    def __getitem__(self,index):
        sample=tuple([self.var[x][index] for x in self.var if self.n_var[x]])
        if self.transform:
            sample=self.transform(sample) 
        return sample
    
    def __len__(self):
        return self.n_samples
    
    def reset(self):
        self.var=self.origin.copy()
        self.sys_array=self.sys_array_origin.copy()
        if self.n_var['U']: self.var['U']=self.prepare_u() 
        self.window={}
        self.type_stats=[]
        
    def prepare_u(self):
        u_list=[]
        for sys in self.sys_list:
            sys_mask=np.isin(self.sys_array,sys)
            sys_ind=self.sys_list.index(sys)
            sys_life=sum(sys_mask)
            for j in range(sys_life):
                u_list.append(np.concatenate((self.origin['U'][sys_ind,j:],self.origin['U'][sys_ind,:j]),axis=0))
        return np.stack(u_list,axis=0)

    def add_window(self,window):
        if not self.window: 
            lists={var:[] for var in window if self.n_var[var]}
            start_index,cut_end=np.max(list(window.values()),axis=0)
            end_index = None if cut_end == 0 else -cut_end
            w_mask=np.zeros_like(self.sys_array, dtype=bool)      
            for sys in self.sys_list:
                sys_mask=np.isin(self.sys_array,sys)
                #start_index,cut_end=np.max([window[0],window[1],window[2]],axis=0)               
                sys_index=np.where(sys_mask)[0][start_index:end_index]
                w_mask[sys_index]=True
                for j in sys_index:
                    for x in lists: 
                        lists[x].append(self.var[x][j-window[x][0]:j+window[x][1]+1,...] ) 
            self.window=window
            self.var.update({x:np.stack(lists[x],axis=0) for x in lists})
            self.var.update({x:self.var[x][w_mask,...] for x in self.var if x not in lists and self.n_var[x]})

            self.n_samples=len(self.var['X'])
            self.sys_array=self.sys_array[w_mask]
        else:
            print('The dataset is already temporized. Use reset method first') #This is for simplicity
    
    def u_sys_separation(self):
        U_sep=sys_separation(self.u,self.u_sys_array)
        sys_sep=sys_separation(self.u_sys_array,self.u_sys_array)
        return U_sep,sys_sep
    

    def sub_collect(self,sys_subset):
        if set(sys_subset).issubset(self.sys_list):
            sys_subset_list=list(sys_subset)
            mask=np.isin(self.sys_array_origin,sys_subset_list)
            mask_u=np.isin(self.sys_list,sys_subset_list)
            
            var=defaultdict(lambda: np.array([]))
            var.update({var:self.origin[var][mask] for var in self.origin if var!='U' and self.n_var[var]})
            if self.n_var['U']: var['U']=self.origin['U'][mask_u]

            sub_dataset=RULDataset_fast(data=(var['X'],var['Z'],var['U'],var['Y']),
                                sys_array=self.sys_array_origin[mask],
                                data_labels=self.labels,
                                data_name=self.data_name,
                                dataset_name=self.dataset_name) 
        else:
            print(f'Choose a sub-selection of the systems of the dataset: {self.sys_list}')
            sub_dataset=None
        return sub_dataset

    def compute_stats(self,norm_type,variables):      
        return {var:compute_stats(self.origin[var],norm_type=norm_type) for var in variables if self.n_var[var]} 

    def normalize(self,norm_type,stats=None):
        if len(self.window)+len(self.type_stats):
            self.reset()
        if isinstance(stats,list):
            stats=self.compute_stats(norm_type=norm_type,variables=stats)
        if isinstance(stats,dict): 
            for x in stats:
                self.var[x]=normalize(self.var[x],norm_type=norm_type,stats=stats[x])
        else:
            print('stats needs to have the form [norm_type,list] for no stats or [norm_type,dict] for providing the stats')
            return None
        self.type_stats=[norm_type,stats]            

    # def temporize_pad(self,window=10):
    #     if not self.window:            
    #         dataset_temp=np.zeros([self.n_samples,window,self.n_features],dtype=self.x.dtype)
    #         for ind in range(self.n_samples):
    #             for look in range(0,window):
    #                 if self.sys_array[ind-look]==self.sys_array[ind]:
    #                     feat=self.x[ind-look,:]
    #                 dataset_temp[ind,window-look-1,:]=feat
    #         self.window=window
    #         self.x=dataset_temp
    #     else:
    #         print('The dataset has a window already. Please reset the dataset first') #This is for simplicity

    
    # def add_scores(self):
    #     if self.scored:
    #         print('Dataset already scored')
    #     else:
    #         mu_y=np.mean(self.y,axis=0)
    #         std_y=np.std(self.y,axis=0)    
    #         std_score=(self.y-mu_y)/std_y      
    #         self.y=np.concatenate((self.y[:,np.newaxis],std_score[:,np.newaxis]), axis=1) 
    #         self.scored=True




# ## Data Loader Creation
# scaling_label is a lambda function
def create_loader(dataset,batch_size,stats,given_stats={},norm_type='minmax',window=None,shuffle=False):
    #reset if necesary
    if len(dataset.window)+len(dataset.type_stats):
        dataset.reset()
    dataset.normalize(norm_type,stats=stats)
    
    #normalize
    stats=dataset.compute_stats(norm_type,variables=stats)
    stats.update(given_stats)
    dataset.normalize(norm_type,stats=stats)
    
    #temporize
    if len(window):
        dataset.add_window(window)


    train_loader=DataLoader(dataset=dataset, batch_size=batch_size,shuffle=shuffle)
    
    return train_loader



def create_loaders(train_dataset,test_dataset,batch_size,stats,given_stats={},window=None,norm_type='minmax'):
    #reset if necesary
    if len(train_dataset.window)+len(train_dataset.type_stats):
        train_dataset.reset()
    if len(test_dataset.window)+len(test_dataset.type_stats):
        test_dataset.reset()
    
    #normalize
    stats=train_dataset.compute_stats(norm_type,variables=stats)
    stats.update(given_stats)
    train_dataset.normalize(norm_type,stats=stats)
    test_dataset.normalize(norm_type,stats=stats)

    #temporize
    if len(window):
        train_dataset.add_window(window)
        test_dataset.add_window(window)
    
    train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size[0],shuffle=True)
    test_loader=DataLoader(dataset= test_dataset, batch_size=batch_size[1],shuffle=False)
    
    return train_loader, test_loader

def create_cross_loaders(dataset,n_folds,batch_size,stats,given_stats={},window=False,shuffle=True,norm_type='minmax'):
    fold_len=dataset.n_sys//n_folds
    sys_list=dataset.sys_list
    
    #Cross validation Shufle
    if shuffle:
        random.shuffle(sys_list)
        
    train_loaders=[]
    val_loaders=[]
    for fold_cut in range(0, dataset.n_sys, fold_len):
        val_list=sys_list[fold_cut:fold_cut+fold_len]
        train_list=list(set(sys_list)-set(val_list))
        train_dataset=dataset.sub_collect(train_list)
        val_dataset=dataset.sub_collect(val_list)        
        
        train_loader, val_loader=create_loaders(train_dataset,val_dataset,
                                                batch_size=(batch_size,batch_size),
                                                stats=stats,
                                                given_stats=given_stats,
                                                window=window,
                                                norm_type=norm_type)
        
        # Build DataLoader
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    
    return train_loaders,val_loaders

def plot_features(features,labels=None):
    if features.ndim == 2:
        features=np.expand_dims(features, axis=2)
    n_sys,max_life,n_fea=features.shape

    t=range(max_life)
    colors=plt.cm.viridis(np.linspace(0, 1, n_sys))
    #colors = plt.cm.viridis(np.linspace(0, 1, n_sys))
    if not labels:
        labels=range(n_fea)
    for fea in range(n_fea):
        plt.title(f'Feature {labels[fea]}') 
        plt.xlabel("time") 
        plt.ylabel('value') 
        for sys in range(n_sys): 
            plt.plot(t,features[sys,:,fea].T,color=colors[sys])
        plt.show()



# plt.cm.viridis
# plt.cm.tab10: Similar to tab20, but with fewer distinct colors.
# plt.cm.jet: A colormap with a rainbow spectrum of colors.
# plt.cm.inferno: A colormap with a fiery spectrum.
# plt.cm.coolwarm: A colormap with cool and warm colors.