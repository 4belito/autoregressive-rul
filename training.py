import torch
import torch.nn as nn
import numpy as np
import metrics as met

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_x_y(model,train_loader,n_epoch,lr,verbose=False):
    # objective
    objective = nn.MSELoss()
    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # training loop
    for epoch in range(n_epoch):
        if verbose:
            loss_epoch = []
        for x, y in train_loader:
            # model1
            x = x.to(device)
            y = y.to(device)  
            y_hat = model(x).reshape(y.shape)
            # perform=y[...,0]
            optim.zero_grad()     
            loss = objective(y_hat, y)
            loss.backward()
            optim.step()
            # model2
            if verbose:
                loss_epoch.append(np.sqrt(loss.item()))
        if verbose:
            loss_epoch = np.mean(np.stack(loss_epoch))
            print(f'----Epoch {epoch}:----')
            print(f'train loss1: {loss_epoch}')
    return model

def evaluate_x_y(model,test_loader,verbose=False):   
    #plot_test logs
    y_pred_test=np.empty([0,*test_loader.dataset.var['Y'].shape[1:]])
    y_true_test=np.empty([0,*test_loader.dataset.var['Y'].shape[1:]])
    
    model.eval()
    if verbose:
        loss=[]
    with torch.no_grad():
        for x,y in test_loader:              
            x = x.to(device)
            y = y.to(device) 

            #model
            y_hat = model(x).reshape(y.shape)

            y_hat_np=y_hat.to('cpu').detach().numpy()
            y_np=y.to('cpu').detach().numpy()
            y_pred_test=np.append(y_pred_test,y_hat_np,axis=0)            
            y_true_test=np.append(y_true_test,y_np,axis=0)
            #
            if verbose:
                loss.append(np.sqrt(np.mean((y_hat_np-y_np)**2)))
    if verbose:
        loss=np.mean(np.stack(loss)) 
        print(f'test loss1: {loss}')

    model.train() # I think we don't need this
    return y_pred_test,y_true_test



def train_xu_y(model,train_loader,n_epoch,lr,verbose=False):
    # objective
    objective = nn.MSELoss()
    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # training loop
    for epoch in range(n_epoch):
        if verbose:
            loss_epoch = []
        for x, u, y in train_loader:
            # model1
            x = x.to(device)
            u = u.to(device) 
            y = y.to(device)  
            y_hat = model(x,u).reshape(y.shape)
            # perform=y[...,0]
            optim.zero_grad()     
            loss = objective(y_hat, y)
            loss.backward()
            optim.step()
            # model2
            if verbose:
                loss_epoch.append(np.sqrt(loss.item()))
        if verbose:
            loss_epoch = np.mean(np.stack(loss_epoch))
            print(f'----Epoch {epoch}:----')
            print(f'train loss1: {loss_epoch}')
    return model

def evaluate_xu_y(model,test_loader,verbose=False):   
    #plot_test logs
    y_pred_test=np.empty([0,*test_loader.dataset.var['Y'].shape[1:]])
    y_true_test=np.empty([0,*test_loader.dataset.var['Y'].shape[1:]])
    
    model.eval()
    if verbose:
        loss=[]
    with torch.no_grad():
        for x,u, y in test_loader:              
            x = x.to(device)
            u = u.to(device) 
            y = y.to(device) 

            #model
            y_hat = model(x,u).reshape(y.shape)

            y_hat_np=y_hat.to('cpu').detach().numpy()
            y_np=y.to('cpu').detach().numpy()
            y_pred_test=np.append(y_pred_test,y_hat_np,axis=0)            
            y_true_test=np.append(y_true_test,y_np,axis=0)
            #
            if verbose:
                loss.append(np.sqrt(np.mean((y_hat_np-y_np)**2)))
    if verbose:
        loss=np.mean(np.stack(loss)) 
        print(f'test loss1: {loss}')

    model.train() # I think we don't need this
    return y_pred_test,y_true_test

def train_xz_y(model,train_loader,n_epoch,lr,verbose=False):
    # objective
    objective = nn.MSELoss()
    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # training loop
    for epoch in range(n_epoch):
        if verbose:
            loss_epoch = []
        for x, z, y in train_loader:
            # model1
            x = x.to(device)
            z = z.to(device) 
            y = y.to(device)  
            x_joint=torch.concatenate((x,z),axis=2)
            y_hat = model(x_joint).reshape(y.shape)
            # perform=y[...,0]
            optim.zero_grad()     
            loss = objective(y_hat, y)
            loss.backward()
            optim.step()
            # model2
            if verbose:
                loss_epoch.append(np.sqrt(loss.detach().item()))
        if verbose:
            loss_epoch = np.mean(np.stack(loss_epoch))
            print(f'----Epoch {epoch}:----')
            print(f'train loss1: {loss_epoch}')
    return model

def evaluate_xz_y(model,test_loader,verbose=False):   
    #plot_test logs
    y_pred_test=np.empty([0,*test_loader.dataset.var['Y'].shape[1:]])
    y_true_test=np.empty([0,*test_loader.dataset.var['Y'].shape[1:]])
    
    model.eval()
    if verbose:
        loss=[]

    for x,z, y in test_loader:              
        x = x.to(device)
        z = z.to(device) 
        y = y.to(device) 

        #model1
        x_joint=torch.concatenate((x,z),axis=2)
        y_hat = model(x_joint).reshape(y.shape)

        y_hat_np=y_hat.to('cpu').detach().numpy()
        y_np=y.to('cpu').detach().numpy()
        y_pred_test=np.append(y_pred_test,y_hat_np,axis=0)            
        y_true_test=np.append(y_true_test,y_np,axis=0)
        #
        if verbose:
            loss.append(np.sqrt(np.mean((y_hat_np-y_np)**2)))

    if verbose:
        loss=np.mean(np.stack(loss)) 
        print(f'test loss1: {loss}')

    model.train() # I think we don't need this
    return y_pred_test,y_true_test


def train_compo(model,perform_model,train_loader,n_epoch,lr,verbose=False):
    # objective
    objective = nn.MSELoss()
    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # training loop
    for epoch in range(n_epoch):
        if verbose:
            loss_epoch = []
        for x, z, u, y in train_loader:
            # model1
            x = x.to(device)
            z = z.to(device)
            u = u.to(device) 
            y = y.to(device)
            x_joint=torch.concatenate((x,z),axis=2)
            with torch.no_grad():
                perform=perform_model(x_joint)
                #plt.plot(range(len(perform)),perform.to('cpu').numpy()[:,0])
                #plt.show()
            y_hat = model(perform,u).reshape(y.shape)
            # perform=y[...,0]
            optim.zero_grad()     
            loss = objective(y_hat, y)
            loss.backward()
            optim.step()
            # model2
            if verbose:
                loss_epoch.append(np.sqrt(loss.item()))
        if verbose:
            loss_epoch = np.mean(np.stack(loss_epoch))
            print(f'----Epoch {epoch}:----')
            print(f'train loss1: {loss_epoch}')
    return model

def evaluate_compo(model,perform_model,test_loader,threshold=0,verbose=False):   
    #plot_test logs
    y_pred_test=np.empty([0,*test_loader.dataset.var['Y'].shape[1:]])
    y_true_test=np.empty([0,*test_loader.dataset.var['Y'].shape[1:]])
    
    model.eval()
    if verbose:
        loss=[]
    with torch.no_grad():
        for x,z,u, y in test_loader:              
            x = x.to(device)
            z = z.to(device)
            u = u.to(device) 
            y = y.to(device) 
            x_joint=torch.concatenate((x,z),axis=2)
            perform=perform_model(x_joint)
            #perform_threshold=torch.clip(perform+threshold,min=-0.2,max=1.2)
            #perform_threshold=perform*threshold
            y_hat =(1+threshold)*model(perform/(1+threshold),u).reshape(y.shape)
            #model
            y_hat_np=y_hat.to('cpu').numpy()
            y_np=y.to('cpu').numpy()
            y_pred_test=np.append(y_pred_test,y_hat_np,axis=0)            
            y_true_test=np.append(y_true_test,y_np,axis=0)
            #
            if verbose:
                loss.append(np.sqrt(np.mean((y_hat_np-y_np)**2)))
    if verbose:
        loss=np.mean(np.stack(loss)) 
        print(f'test loss1: {loss}')

    model.train() # I think we don't need this
    return y_pred_test,y_true_test


def evaluate_diff(model,perform_model,test_loader,threshold=0,verbose=False):   
    #plot_test logs
    y_pred_test=np.empty([0,*test_loader.dataset.var['Y'].shape[1:]])
    y_true_test=np.empty([0,*test_loader.dataset.var['Y'].shape[1:]])
    
    model.eval()
    if verbose:
        loss=[]
    with torch.no_grad():
        for x,z,u, y in test_loader:              
            x = x.to(device)
            z = z.to(device)
            u = u.to(device) 
            y = y.to(device) 
            x_joint=torch.concatenate((x,z),axis=2)
            perform=perform_model(x_joint)
            #perform_threshold=torch.clip(perform+threshold,min=-0.2,max=1.2)
            #perform_threshold=perform*threshold
            y_hat =model(perform-threshold,u).reshape(y.shape)
            #model
            y_hat_np=y_hat.to('cpu').numpy()
            y_np=y.to('cpu').numpy()
            y_pred_test=np.append(y_pred_test,y_hat_np,axis=0)            
            y_true_test=np.append(y_true_test,y_np,axis=0)
            #
            if verbose:
                loss.append(np.sqrt(np.mean((y_hat_np-y_np)**2)))
    if verbose:
        loss=np.mean(np.stack(loss)) 
        print(f'test loss1: {loss}')

    model.train() # I think we don't need this
    return y_pred_test,y_true_test
















def train_old_compo(model1, model2, train_loader,n_epoch,lr1,lr2,window,verbose=False):
    # objective
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    objective1 = nn.MSELoss()
    objective2 = nn.MSELoss()
    # optimizer
    optim1 = torch.optim.Adam(model1.parameters(), lr=lr1)
    optim2 = torch.optim.Adam(model2.parameters(), lr=lr2)
    # training loop
    for epoch in range(n_epoch):
        if verbose:
            loss1_epoch = []
            loss2_epoch = []
        for x, z, u, y in train_loader:
            # model1
            x = x.to(device)
            z = z.to(device)
            u = u.to(device)
            y = y.to(device)      
            if window[0][1] > 0:
                x = x[:, :window[0][0]+1, :]
            z_hat = model1(x)
            # perform=y[...,0]
            optim1.zero_grad()     
            loss1 = objective1(z_hat, z)
            loss1.backward()
            optim1.step()
            # model2
            y_hat = model2(z, u)
            optim2.zero_grad()     
            loss2 = objective2(y_hat, y)
            loss2.backward()
            optim2.step()
            if verbose:
                loss1_epoch.append(np.sqrt(loss1.item()))
                loss2_epoch.append(np.sqrt(loss2.item()))
        if verbose:
            loss1_epoch = np.mean(np.stack(loss1_epoch))
            loss2_epoch = np.mean(np.stack(loss2_epoch))   
            print(f'----Epoch {epoch}:----')
            print(f'train loss1: {loss1_epoch}')
            print(f'train loss2: {loss2_epoch}')
    return model1, model2 




def evaluate_old_compo(model1,model2,test_loader,window,verbose=False):         
    #plot_test logs
    z_pred_test=np.empty([0,window[1][1]+1]).squeeze()
    z_true_test=np.empty([0,window[1][1]+1]).squeeze()
    y_pred_test=np.empty(0)
    y_true_test=np.empty(0)
    
    model1.eval()
    model2.eval()
    if verbose:
        loss1=[]
        loss2=[]
    with torch.no_grad():
        for x,z,u, y in test_loader:              
            x = x.to(device)
            z = z.to(device)
            u = u.to(device)
            y = y.to(device) 

            #model1
            if window[0][1]>0:
                x=x[:,:window[0][0]+1,:]
            z_hat = model1(x)

            z_hat_np=z_hat.to('cpu').detach().numpy()
            z_np=z.to('cpu').detach().numpy()
            z_pred_test=np.append(z_pred_test,z_hat_np,axis=0)            
            z_true_test=np.append(z_true_test,z_np,axis=0)
            #
            
            #model2
            y_hat = model2(z_hat,u)
            
            y_hat_np=y_hat.to('cpu').detach().numpy()
            y_np=y.to('cpu').detach().numpy()
            y_pred_test=np.append(y_pred_test,y_hat_np,axis=0)            
            y_true_test=np.append(y_true_test,y_np,axis=0)
            if verbose:
                loss1.append(np.sqrt(np.mean((z_hat_np-z_np)**2)))
                loss2.append(np.sqrt(np.mean((y_hat_np-y_np)**2)))

    if verbose:
        loss1=np.mean(np.stack(loss1))
        loss2=np.mean(np.stack(loss2))   
        print(f'test loss1: {loss1}')
        print(f'test loss2: {loss2}')

    model1.train()
    model2.train()
    return z_pred_test,z_true_test,y_pred_test,y_true_test



def train_autoencoder(model, train_loader,n_epoch,lr,verbose=False):
    # objective
    objective = nn.MSELoss()
    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # training loop
    for epoch in range(n_epoch):
        if verbose:
            loss_epoch = []
        for x, _, _, _ in train_loader:
            # model1
            x = x.to(device)     
            x_hat = model(x)
            # perform=y[...,0]
            optim.zero_grad()     
            loss = objective(x_hat, x)
            loss.backward()
            optim.step()
            # model2
            if verbose:
                loss_epoch.append(np.sqrt(loss.item()))
        if verbose:
            loss_epoch = np.mean(np.stack(loss_epoch))
            print(f'----Epoch {epoch}:----')
            print(f'train loss1: {loss_epoch}')
    return model


def evaluate_autoencoder(model,test_loader,verbose=False):   
    #plot_test logs
    x_pred_test=np.empty([0,test_loader.dataset.x.shape[1]])
    x_true_test=np.empty([0,test_loader.dataset.x.shape[1]])
    
    model.eval()
    if verbose:
        loss=[]
    with torch.no_grad():
        for x,_,_, _ in test_loader:              
            x = x.to(device)

            #model1
            x_hat = model(x)

            x_hat_np=x_hat.to('cpu').detach().numpy()
            x_np=x.to('cpu').detach().numpy()
            x_pred_test=np.append(x_pred_test,x_hat_np,axis=0)            
            x_true_test=np.append(x_true_test,x_np,axis=0)
            #
            if verbose:
                loss.append(np.sqrt(np.mean((x_hat_np-x_np)**2)))

    if verbose:
        loss=np.mean(np.stack(loss)) 
        print(f'test loss1: {loss}')

    model.train()
    return x_pred_test,x_true_test






def train_compo_complex(model1,model2,train_loader,config,window=[[0,0],[0,0]],split=[7,2]):

    assert sum(split)==train_loader.dataset.x.shape[-1], 'Incorrect split. Check the number of inputs and variables sum the number of features'

    #objective
    objective = nn.MSELoss()
    
    #optimizer
    optim1 = torch.optim.Adam(model1.parameters(), lr=config['lr1'])

    optim2 = torch.optim.Adam(model2.parameters(), lr=config['lr2'])
    
    #training loop
    for _ in range(config['n_epoch']):
        for x, y in train_loader:
            #model1
            x= x.to(device)
            y= y.to(device)      
            fea,input=torch.split(x, split,dim=-1)
            if window[0][1]>0:
                fea=fea[:,:window[0][0]+1,:]
            x1=torch.cat((fea.flatten(1),input.flatten(1)),dim=1)
            perform=y[...,0]
            y1_hat = model1(x1)

            optim1.zero_grad()     
            loss1 = objective(y1_hat,perform)
            loss1.backward()
            optim1.step()
            #print(f'train loss1: {loss1.item()}')
            #model2
            #_,input=torch.split(input,[window[0][0],1+window[0][1]],dim=1)
            if window[0][0]>0:
                input=input[:,window[0][0]:,:].squeeze(1)

            x2=torch.cat((perform.unsqueeze(-1),input),dim=-1)
            if sum(window[1])>0:
                y2=y[:,0,1]
            else:
                y2=y[:,1]
            y2_hat = model2(x2)

            optim2.zero_grad()     
            loss2 = objective(y2_hat,y2)
            loss2.backward()
            optim2.step()
            #print(f'train loss2: {loss2.item()}')
    return model1,model2 



def evaluate_compo_complex(model1,model2,test_loader,window=[[0,0],[0,0]],split=[7,2]):           
    #plot_test logs
    y_pred1_test=np.empty([0,window[0][1]+1]).squeeze()
    y_true1_test=np.empty([0,window[0][1]+1]).squeeze()
    y_pred2_test=np.empty(0)
    y_true2_test=np.empty(0)
    
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for x, y in test_loader:              
            x = x.to(device)
            y = y.to(device) 
            
            #model1
            fea,input=torch.split(x, split,dim=-1)
            if sum(window[0])>0:
                fea=fea[:,:window[0][0]+1,:]
            x1=torch.cat((fea.flatten(1),input.flatten(1)),dim=1)
            perform_hat = model1(x1)

            perform_hat_np=perform_hat.to('cpu').detach().numpy()
            y1_np=y[...,0].to('cpu').detach().numpy()
            y_pred1_test=np.append(y_pred1_test,perform_hat_np,axis=0)            
            y_true1_test=np.append(y_true1_test,y1_np,axis=0)
            #print(f'test loss1: {np.mean((perform_hat_np-y_np)**2)}')
            
            #model2
            if window[0][0]>0:
                input=input[:,window[0][0]:,:].squeeze(1)
            x=torch.cat((perform_hat.unsqueeze(-1),input),dim=-1)
            if sum(window[1])>0:
                y=y[:,0,1]
            else:
                y=y[:,1]
            y_hat = model2(x)
            
            y_hat_np=y_hat.to('cpu').detach().numpy()
            y_np=y.to('cpu').detach().numpy()
            y_pred2_test=np.append(y_pred2_test,y_hat_np,axis=0)            
            y_true2_test=np.append(y_true2_test,y_np,axis=0)
            #print(f'test loss1: {np.mean((y_hat-y)**2)}')
    model1.train()
    model2.train()
    return y_pred1_test,y_true1_test,y_pred2_test,y_true2_test


