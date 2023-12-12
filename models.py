import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self,
                input_dim=9,
                embed_dim=9,
                dim_feedforward=2048,
                n_timesteps=10,
                n_head=1,
                n_layers=1,
                dropout=0.5,
                activation="gelu",
                device='cpu'): #128,256
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim*n_timesteps,embed_dim)
        self.positional_embed = nn.Parameter(torch.randn(embed_dim)).to(device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head,dim_feedforward=dim_feedforward, dropout=dropout,activation=activation, batch_first=True)
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers) 
        self.latent_head = nn.Sequential(nn.Linear(embed_dim, dim_feedforward),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(dim_feedforward, dim_feedforward),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(dim_feedforward, embed_dim))
    
    def forward(self, x):
        x = x.flatten(1)
        z = self.input_projection(x)    
        z = self.positional_embed + z   
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
    

class StateTransition(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.transition_fn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, z, n_steps):
        for step_num in range(n_steps):
            z = self.transition_fn(z)
            
        return z
    
    
class RULEstimator(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.25):
        super().__init__()
        self.eol_estimation = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        pred = self.eol_estimation(z)

        return pred