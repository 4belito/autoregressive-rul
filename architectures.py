import torch
import torch.nn as nn
import math

class RUL_estimation(nn.Module):
    def __init__(self,prediction_window=1,
                look_back=10,
                n_features=8,
                embed_dim=9,
                rul_head_dim=4,
                dim_feedforward=2048,
                n_head=1,
                activation="gelu"): #128,256
        super().__init__()
        
        # Projection into the laten space layer
        self.input_projection = nn.Sequential(nn.Flatten(1),nn.Linear(look_back*n_features,embed_dim))
        
        # posisional embedding parameters
        self.positional_embed = nn.Parameter(torch.randn(embed_dim))
        
        # encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head,dim_feedforward=dim_feedforward, activation=activation)
        
        #Transformer
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=1) #stack of only 1 encoder layer
        
        # 
        self.rul_head = nn.Sequential(nn.Linear(embed_dim, rul_head_dim),
                                        nn.ReLU(),
                                        nn.Linear(rul_head_dim, rul_head_dim),
                                        nn.ReLU(),
                                        nn.Linear(rul_head_dim, prediction_window))
    
    def forward(self, x):     
        z = self.input_projection(x)
        z = self.positional_embed +z            
        z = self.transformer_blocks(z)
        z = self.rul_head(z)
        return z.squeeze(1)


class RUL_transition(nn.Module):
    def __init__(self,prediction_window=1,
                look_back=10,
                n_features=8,
                embed_dim=9,
                rul_head_dim=4,
                dim_feedforward=2048,
                n_head=1,
                activation="gelu"): #128,256
        super().__init__()
        
        # Projection into the laten space layer
        self.input_projection = nn.Sequential(nn.Flatten(1),nn.Linear(look_back*n_features,embed_dim))
        
        # posisional embedding parameters
        self.positional_embed = nn.Parameter(torch.randn(embed_dim))
        
        # encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head,dim_feedforward=dim_feedforward, activation=activation)
        
        #Transformer
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=1) #stack of only 1 encoder layer
        
        # 
        self.rul_head = nn.Sequential(nn.Linear(embed_dim, rul_head_dim),
                                        nn.ReLU(),
                                        nn.Linear(rul_head_dim, rul_head_dim),
                                        nn.ReLU(),
                                        nn.Linear(rul_head_dim, n_features))
    
    def forward(self, x):     
        z = self.input_projection(x)
        z = self.positional_embed +z            
        z = self.transformer_blocks(z)
        z = self.rul_head(z)
        return z.squeeze(1)


class Aggregation(nn.Module):
    def __init__(self,capacitor_model,prediction_window=1,
                look_back=10,
                n_features=8,
                n_capacitors=3,
                agreg_dim=[512,512]): #128,256
    
        super().__init__()
        
        self.prediction_window = prediction_window     
        self.windows = look_back
        self.features = n_features
        self.n_capacitors = n_capacitors #128        
        self.capacitor_model = capacitor_model

        self.agregation_MLP = nn.Sequential(nn.Linear(self.n_capacitors, agreg_dim[0]),
                                        nn.ReLU(),
                                        nn.Linear(agreg_dim[0], agreg_dim[1]),
                                        nn.ReLU(),
                                        nn.Linear(agreg_dim[1], self.prediction_window))
    
    def forward(self, x):    
        RUL1=self.capacitor_model(x[:,:,[0,3,4,5]]).unsqueeze(1)        
        RUL2=self.capacitor_model(x[:,:,[1,3,4,6]]).unsqueeze(1)
        RUL3=self.capacitor_model(x[:,:,[2,3,4,7]]).unsqueeze(1)
        z=torch.cat((RUL1, RUL2, RUL3), 1)
        z = self.agregation_MLP(z)
        return z.squeeze(1)
    
class Aggregation_full(nn.Module):
    def __init__(self,capacitor_model,prediction_window=1,
                look_back=10,
                n_features=8,
                n_capacitors=3,
                agreg_dim=[512,512]): #128,256

        super().__init__()
        
        self.prediction_window = prediction_window     
        self.windows = look_back
        self.features = n_features
        self.n_capacitors = n_capacitors #128        
        self.capacitor_model = capacitor_model

        self.agregation_MLP = nn.Sequential(nn.Linear(self.n_capacitors, agreg_dim[0]),
                                            nn.ReLU(),
                                            nn.Linear(agreg_dim[0], agreg_dim[1]),
                                            nn.ReLU(),
                                            nn.Linear(agreg_dim[1], self.prediction_window))
        
    def forward(self, x):    
        RUL1=self.capacitor_model(x[:,:,[0,1,2,3,4,5,6,7,8,9]]).unsqueeze(1)        
        RUL2=self.capacitor_model(x[:,:,[2,0,1,3,4,5,6,9,7,8]]).unsqueeze(1)
        RUL3=self.capacitor_model(x[:,:,[1,2,0,3,4,5,6,8,9,7]]).unsqueeze(1)
        z=torch.cat((RUL1, RUL2, RUL3), 1)
        z = self.agregation_MLP(z)
        return z.squeeze(1)


    
    
class MLP(nn.Module):
    def __init__(self,prediction_window=1,
                    n_features=3,
                    agreg_dim=[512,512]): #128,256
        
        super().__init__()
        
        self.prediction_window = prediction_window     
        self.features = n_features
        self.agregation_MLP = nn.Sequential(nn.Linear(self.features, agreg_dim[0]),
                                            nn.ReLU(),
                                            nn.Linear(agreg_dim[0], agreg_dim[1]),
                                            nn.ReLU(),
                                            nn.Linear(agreg_dim[1], self.prediction_window))
    
    def forward(self, x):    
        z = self.agregation_MLP(x)
        return z.squeeze(1)
        
        
    
    
class RULTransformer(nn.Module):
    def __init__(self,
                look_back,#=10
                n_features,#=8
                prediction_window=1,
                embed_dim=64,
                #dim_feedforward=4,
                rul_head_dim=512,
                n_head=4,
                activation="gelu"): #128,256
    
        super().__init__()
                    
        
        # Projection into the laten space layer
        self.input_projection = nn.Sequential(nn.Flatten(1),nn.Linear(look_back*n_features, embed_dim))
        
        # posisional embedding parameters
        self.positional_embed = nn.Parameter(torch.randn(embed_dim))
        
        # encoder layer
        #self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, n_head=n_head,dim_feedforward=dim_feedforward, activation=activation)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, n_head=n_head, activation=activation)
        
        #Transformer
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=1) #stack of only 1 encoder layer
        
        # 
        self.rul_head = nn.Sequential(nn.Linear(embed_dim, rul_head_dim),
                                        nn.ReLU(),
                                        nn.Linear(rul_head_dim, rul_head_dim),
                                        nn.ReLU(),
                                        nn.Linear(rul_head_dim, prediction_window))
    
    def forward(self, x):     
        z = self.input_projection(x)
        z = self.positional_embed +z            
        z = self.transformer_blocks(z)
        z = self.rul_head(z)
        return z.squeeze(1)

class RULBiLSTM(nn.Module):
    def __init__(self,n_features,look_back=10,):
        super().__init__()
        
        self.prediction_window = 1
        self.windows = look_back
        self.features = n_features
        #self.embed_dim = embed_dim #128  
        #self.hidden_dim=32
        
        self.lstm = nn.LSTM(input_size=self.features, hidden_size=128, num_layers=3, bidirectional=False, dropout=0, batch_first=True)
        self.dropout=nn.Dropout(p=0.5)
        self.linear=nn.Linear(self.windows*self.features, 1)
        
        
    
    def forward(self, x): 
        # z=self.input_projection(x)
        z=self.lstm(x)[0]
        z = torch.flatten(x, 1)
        # z=self.dropout(z)
        z=self.linear(z)
        return z
    


class EncoderMLP(nn.Module):
    def __init__(self, 
                in_dim=[30,9],
                embedding_dim=10, 
                #dim=[300,250,200,150,100,50]
                layers=[300,200,100]
                ):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Flatten(1),
            nn.Linear(math.prod(in_dim),layers[0]),          
            nn.ReLU(),
            nn.Linear(layers[0],layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1],layers[2]),
            nn.ReLU(),
            # nn.Linear(dim[2],dim[3]),
            # nn.ReLU(),
            # nn.Linear(dim[3],dim[4]),
            # nn.ReLU(),
            # nn.Linear(dim[4],dim[5]),
            # nn.ReLU(),
            # nn.Linear(dim[5],embedding_dim),
            nn.Linear(layers[2],embedding_dim),
        )    

    def forward(self, x):
        return self.encoder(x) 
    
class DecoderMLP(nn.Module):
    def __init__(self, 
                out_dim=[30,9],
                embedding_dim=10,
                layers=[100,200,300]
                ):
        self.out_dim=out_dim
        super().__init__()
        self.decoder=nn.Sequential(
            nn.Linear(embedding_dim,layers[0]),            
            nn.ReLU(),
            nn.Linear(layers[0],layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1],layers[2]),
            nn.ReLU(),
            # nn.Linear(dim[2],dim[3]),
            # nn.ReLU(),
            # nn.Linear(dim[3],dim[4]),
            # nn.ReLU(),
            # nn.Linear(dim[4],dim[5]),
            # nn.ReLU(),
            # nn.Linear(dim[5],math.prod(out_dim)),
            nn.Linear(layers[2],math.prod(out_dim)),
        )
        
        
    def forward(self, x):
        return self.decoder(x).reshape(-1,*self.out_dim) 
    

class AutoencoderMLP(nn.Module):
    def __init__(self, 
                dim=[3],
                layers=[100,200,300],
                emb_dim=2
                ):
        super().__init__()
        self.encoder = EncoderMLP(in_dim=dim,layers=layers[::-1],embedding_dim=emb_dim)
        self.decoder = DecoderMLP(out_dim=dim,layers=layers,embedding_dim=emb_dim)
                                        
    def forward(self, x):
        z=self.encoder(x)
        return self.decoder(z)