import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self,
                input_dim=9,
                embed_dim=9,
                dim_feedforward=2048,
                n_timesteps=10,
                n_head=1,
                dropout=0.5,
                activation="gelu",
                device='cpu'): #128,256
        super().__init__()
        
        self.positional_embed = nn.Parameter(torch.randn(input_dim)).to(device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_head,dim_feedforward=dim_feedforward, dropout=dropout,activation=activation, batch_first=True)
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=1) 
        self.latent_head = nn.Sequential(nn.Linear(input_dim*n_timesteps, dim_feedforward),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(dim_feedforward, dim_feedforward),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(dim_feedforward, embed_dim))
    
    def forward(self, x):    
        z = self.positional_embed + x   
        z = self.transformer_blocks(z)
        z = z.flatten(1)
        z = self.latent_head(z)
        return z
    
class TransformerDecoder(nn.Module):
    def __init__(self,
                output_dim=9,
                embed_dim=9,
                dim_feedforward=2048,
                n_head=1,
                dropout=0.5,
                activation="gelu"): #128,256
        super().__init__()
        
        self.latent_head = nn.Sequential(nn.Linear(embed_dim, dim_feedforward),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(dim_feedforward, dim_feedforward),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(dim_feedforward, embed_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head,dim_feedforward=dim_feedforward, dropout=dropout,activation=activation)
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=1) 
        self.output_projection = nn.Sequential(nn.Linear(embed_dim, dim_feedforward),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(dim_feedforward, dim_feedforward),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(dim_feedforward, output_dim))
    
    def forward(self, x):     
        z = self.latent_head(x)           
        z = self.transformer_blocks(z)
        xhat = self.output_projection(z)
        return xhat