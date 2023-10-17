import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import metrics as met
#import architectures as arc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            diff_scoring=met.DiffScoring(a=self.a,b=self.b,war=self.war,alpha=self.alpha)
            if self.weighted:
                res=diff_scoring.weighted(y_p,y_t)
            else:
                res=diff_scoring(y_p,y_t)
                
        if self.loss=='diff2Scoring':
            diff2scoring=met.Diff2Scoring(a=self.a,b=self.b,war=self.war,alpha=self.alpha)
            if self.weighted:
                res=diff2scoring.weighted(y_p,y_t)
            else:
                res=diff2scoring(y_p,y_t)
                
        if self.loss=='spherical':
            spherical=met.Spherical()
            res=spherical(y_p,y_t)               

        return torch.mean(res)                         


# ## Training Loop
# 1. **model:** model to be trained 
# 2. **config:** configuration of the training
# 3. **train_loader:** Dataloaders containing the data to learn on


def train_estimation(model,train_loader,config,noise_coef=False):
    #loss function
    loss=config['Floss']         

    #objective
    objective = RUL_Loss(config['Floss'],a=config['a'],b=config['b'],war=config['war'],alpha=config['alpha'],weighted=config['weighted'])
    
    #optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    #training loop
    for _ in range(config['n_epoch']):
        for x_data, y_data in train_loader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            x=x_data[:,:-1,:]
            y=y_data            
            
            if noise_coef:
                y_score=y[:,1]
                noise_strength=noise_coef[0]*(y_score>0)+noise_coef[2]*(y_score<0)
                noise_bound=noise_coef[1]*(y_score>0)+noise_coef[3]*(y_score<0)
                noise=noise_strength*torch.clip(torch.abs(y_score)-noise_bound,0)*torch.randn(y.shape[0]).to(device)                         
                y=y[:,0]+torch.sign(y_score)*torch.abs(noise)          
            
            optim.zero_grad()     
            y_hat = model(x)

            loss = objective(y_hat, y.float())
            
            loss.backward()

            optim.step()
        #print(f' loss in epoch {m}: {loss}')        
    return model #

def train_transition(model,train_loader,config,noise_coef=False):
    #loss function
    loss=config['Floss']         

    #objective
    objective = RUL_Loss(config['Floss'],a=config['a'],b=config['b'],war=config['war'],alpha=config['alpha'],weighted=config['weighted'])
    
    #optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    #training loop
    for _ in range(config['n_epoch']):
        for x_data, _ in train_loader:
            x_data = x_data.to(device)
            x=x_data[:,:-1,:]
            y=x_data[:,-1,:]              
            optim.zero_grad()     
            y_hat = model(x)

            loss = objective(y_hat, y.float())
            
            loss.backward()

            optim.step()
        #print(f' loss in epoch {m}: {loss}')        
    return model #



def train_autoencoder(model,train_loader,config):
    optimizer = torch.optim.Adam(params=model.parameters(),lr=config['lr'])

    for _ in range(config['n_epoch']):
        for inputs,_ in train_loader:
            x=inputs[:,:-1,:].to(device)
            recon_x=model(x)            
            loss=F.mse_loss(recon_x, x)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def evaluate_estimation(model,test_loader):           
    #plot_test logs
    y_pred_test=np.empty(0)
    y_true_test=np.empty(0)
    
    model.eval()
    with torch.no_grad():
        for x_data, y_data in test_loader:              
            x_data = x_data.to(device)
            x=x_data[:,:-1,:]
            #prediction                
            y_hat = model(x) 
            #plot logs
            y_pred_test=np.append(y_pred_test,y_hat.to('cpu').detach().numpy(),axis=0)            
            y_true_test=np.append(y_true_test,y_data.detach().numpy(),axis=0)
    model.train()
    return y_pred_test,y_true_test

def evaluate_transition(model,test_loader):           
    #plot_test logs
    n_features=test_loader.dataset.n_features
    y_pred_test=np.empty([0,n_features])
    y_true_test=np.empty([0,n_features])
    
    model.eval()
    with torch.no_grad():
        #for i, (x, y) in enumerate(tqdm(test_loader[fold])):
        for (x_data, y_data) in test_loader:              
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            x=x_data[:,:-1,:]
            y=x_data[:,-1,:]
            #prediction                
            y_hat = model(x) 
            #plot logs
            y_pred_test=np.append(y_pred_test,y_hat.to('cpu').detach().numpy(),axis=0)            
            y_true_test=np.append(y_true_test,y.to('cpu').detach().numpy(),axis=0)
    model.train()
    return y_pred_test,y_true_test

def evaluate(est_model,tra_model,test_loader):           
    #plot_test logs
    n_features=test_loader.dataset.n_features
    y_pred_rec_test=np.empty(0)
    y_pred_rul_test=np.empty(0)
    y_true_test=np.empty(0)
    
    est_model.eval()
    tra_model.eval()
    with torch.no_grad():
        #for i, (x, y) in enumerate(tqdm(test_loader[fold])):
        for x, y in test_loader: 
            x =x[:,:-1].to(device)
            rul=0
            rul_est=est_model(x).item()
            y_pred_rul_test=np.append(y_pred_rul_test,rul_est.to('cpu').detach().numpy(),axis=0) 
            while rul_est>0:
                x_next = tra_model(x)
                x=torch.cat((x[:,1:],x_next.unsqueeze(1)),axis=1)
                rul_est=est_model(x).item()
                rul+=1               
            #plot logs
            y_pred_rec_test=np.append(y_pred_rec_test,rul,axis=0)            
            y_true_test=np.append(y_true_test,y.detach().numpy(),axis=0)
    return y_pred_rec_test,y_pred_rul_test,y_true_test


def evaluate_autoregressive(EOL_model,tra_model,test_loader,threshold):           
    y_pred=[]
    EOL_model.eval()
    tra_model.eval()
    with torch.no_grad():
        i=0
        #for x, _ in tqdm(test_loader): 
        for x, _ in test_loader: 
            x =x[:,:-1].to(device)
            rul=0
            rul_predictions = np.ones(x.shape[0]) * -1
            #y_est=[]
            while np.count_nonzero(rul_predictions<0)>0:
                x_next = tra_model(x)
                x=torch.cat((x[:,1:],x_next.unsqueeze(1)),axis=1)
                recon_x=EOL_model(x)
                recon_error=torch.mean((recon_x-x)**2,dim=(1,2)).cpu().detach().numpy() 
                not_done=rul_predictions<0
                #print(recon_error)
                rul_predictions[(recon_error >threshold) & not_done ] = rul    
                rul+=1
                #print(rul)
                if rul>300:
                    rul_predictions[not_done] = rul               
            y_pred.append(rul_predictions)
            #y_pred_full
            # if i >0:
            #     break
            # i+=1
    y_pred=np.concatenate(y_pred)
    return y_pred


def evaluate_autoencoder(model,test_loader):           
    y_pred=[]
    y_true=[]
    model.eval()
    with torch.no_grad():
        for x, _ in test_loader: 
            x =x[:,:-1].to(device)
            recon_x = model(x).cpu().detach().numpy()
            y_pred.append(recon_x)
            y_true.append(x.cpu().detach().numpy())
    y_pred=np.concatenate(y_pred)
    y_true=np.concatenate(y_true)
    return y_pred,y_true