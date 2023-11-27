import torch
import torch.nn as nn

class RUL_estimation(nn.Module):
    def __init__(self,prediction_window=1,
                fea_window=10,
                n_features=8,
                embed_dim=9,
                rul_head_dim=4,
                dim_feedforward=2048,
                n_head=1,
                activation="gelu"): #128,256
        super().__init__()
        
        # Projection into the laten space layer
        self.input_projection = nn.Sequential(nn.Flatten(1),nn.Linear(fea_window*n_features,embed_dim))
        
        # positional embedding parameters
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

class Transformer_light(nn.Module):
    def __init__(self,
                feat_dim=9,
                out_dim=1,
                mlp_dims=[64,32,16],
                dim_feedforward=2048,
                n_head=1,
                n_layers=3,
                activation="gelu"): #128,256
        super().__init__()
        # Projection into the laten space layer
        embed_dim=feat_dim*n_head

        self.input_projection = nn.Sequential(nn.Linear(feat_dim,embed_dim))
        
        # posisional embedding parameters
        self.positional_embed = nn.Parameter(torch.randn(embed_dim))
        
        # encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head,dim_feedforward=dim_feedforward, activation=activation,batch_first=True)
        
        #Transformer
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers) #stack of 3 encoder layer
        
        # 
        self.MLP_head = nn.Sequential(nn.Linear(embed_dim, mlp_dims[0]),
                                        nn.ReLU(),
                                        nn.Linear(mlp_dims[0], mlp_dims[1]),
                                        nn.ReLU(),
                                        nn.Linear(mlp_dims[1], mlp_dims[2]),
                                        nn.ReLU(),
                                        nn.Linear(mlp_dims[2], out_dim))
    
    def forward(self, x):     
        x_embed = self.input_projection(x)
        z = self.positional_embed +x_embed            
        z = self.transformer_blocks(z)
        out = self.MLP_head(z)
        return out.squeeze(1)


class Transformer_enc(nn.Module):
    def __init__(self,
                input_dim=9,
                pred_window=1,
                embed_dim=9,
                head_dim=4,
                dim_feedforward=2048,
                n_head=1,
                dropout=0.5,
                activation="gelu"): #128,256
        super().__init__()
        # Projection into the laten space layer
        self.input_projection = nn.Sequential(nn.Flatten(1),nn.Linear(input_dim,embed_dim))
        
        # positional embedding parameters
        self.positional_embed = nn.Parameter(torch.randn(embed_dim))

        # encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head,dim_feedforward=dim_feedforward, dropout=dropout,activation=activation)
        
        #Transformer
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=1) #stack of only 1 encoder layer
        
        # 
        self.rul_head = nn.Sequential(nn.Linear(embed_dim, head_dim),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(head_dim, head_dim),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(head_dim, pred_window))
    
    def forward(self, x):     
        z = self.input_projection(x)
        z = self.positional_embed +z            
        z = self.transformer_blocks(z)
        z = self.rul_head(z)
        return z.squeeze(1)
    

class Transformer_01(nn.Module):
    def __init__(self,
                input_dim=9,
                pred_window=1,
                embed_dim=9,
                head_dim=4,
                dim_feedforward=2048,
                n_head=1,
                dropout=0.5,
                activation="gelu"): #128,256
        super().__init__()
        # Projection into the laten space layer
        self.input_projection = nn.Sequential(nn.Flatten(1),nn.Linear(input_dim,embed_dim))
        
        # positional embedding parameters
        self.positional_embed = nn.Parameter(torch.randn(embed_dim))

        # encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head,dim_feedforward=dim_feedforward, dropout=dropout,activation=activation)
        
        #Transformer
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=1) #stack of only 1 encoder layer
        
        # 
        self.rul_head = nn.Sequential(nn.Linear(embed_dim, head_dim),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(head_dim, head_dim),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(head_dim, pred_window),
                                        nn.Sigmoid())
    
    def forward(self, x):     
        z = self.input_projection(x)
        z = self.positional_embed +z            
        z = self.transformer_blocks(z)
        z = self.rul_head(z)
        return z.squeeze(1)


class Transformer_old(nn.Module):
    def __init__(self,
                input_dim=9,
                pred_window=1,
                embed_dim=9,
                head_dim=4,
                dim_feedforward=2048,
                n_head=1,
                activation="gelu"): #128,256
        super().__init__()
        # Projection into the laten space layer
        self.input_projection = nn.Sequential(nn.Flatten(1),nn.Linear(input_dim,embed_dim))
        
        # positional embedding parameters
        self.positional_embed = nn.Parameter(torch.randn(embed_dim))

        # encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head,dim_feedforward=dim_feedforward, activation=activation)
        
        #Transformer
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=1) #stack of only 1 encoder layer
        
        # 
        self.rul_head = nn.Sequential(nn.Linear(embed_dim, head_dim),
                                        nn.ReLU(),
                                        nn.Linear(head_dim, head_dim),
                                        nn.ReLU(),
                                        nn.Linear(head_dim, pred_window))
    
    def forward(self, x):     
        z = self.input_projection(x)
        z = self.positional_embed +z            
        z = self.transformer_blocks(z)
        z = self.rul_head(z)
        return z.squeeze(1)

class RUL_Transformer(nn.Module):
    def __init__(self,
                input_dim=9,
                embed_dim=9,
                head_dim=4,
                dim_feedforward=2048,
                n_head=1,
                dropout=0.2,
                mlp_dim=[128,128],
                u_enc_dim=9,
                perform_w=1,              
                dropout_pred=0.2,
                pred_window=1,
                ): 
        super().__init__()
        
        # Projection into the laten space layer
        self.inputs_encoder=Transformer_enc(input_dim=input_dim,
                                            pred_window=u_enc_dim,
                                            embed_dim=embed_dim,
                                            head_dim=head_dim,
                                            dim_feedforward=dim_feedforward,
                                            n_head=n_head,
                                            dropout=dropout)
        
       
        self.rul_head = nn.Sequential(nn.Flatten(1),nn.Linear(u_enc_dim+perform_w, mlp_dim[0]),
                                nn.ReLU(),
                                nn.Dropout(p=dropout_pred),
                                nn.Linear(mlp_dim[0], mlp_dim[1]),
                                nn.ReLU(),
                                nn.Dropout(p=dropout_pred),
                                nn.Linear(mlp_dim[1], pred_window))                                      
                                        
    
    def forward(self, x,u):     
        u_enc=self.inputs_encoder(u)
        x=x.reshape(x.shape[0],-1)
        u_enc=u_enc.reshape(u_enc.shape[0],-1)
        x=torch.concatenate((x,u_enc),axis=1)
        z = self.rul_head(x)
        return z.squeeze(1)

class end2end_compo_full(nn.Module):
    def __init__(self,
                input_dim_enc=9,
                pred_window_enc=1,
                embed_dim_enc=9,
                head_dim_enc=4,
                dim_feedforward_enc=2048,
                n_head_enc=1,
                dropout_enc=0.5,
                input_dim_rul=9,
                embed_dim_rul=9,
                head_dim_rul=4,
                dim_feedforward_rul=2048,
                n_head_rul=1,
                dropout_rul=0.2,
                mlp_dim_rul=[128,128],
                u_enc_dim_rul=9,# perform_w_rul=1,              
                dropout_pred_rul=0.2,
                pred_window_rul=1,
                ): 
        super().__init__()

        self.transformer_enc=Transformer_enc(input_dim=input_dim_enc,
                pred_window=pred_window_enc,
                embed_dim=embed_dim_enc,
                head_dim=head_dim_enc,
                dim_feedforward=dim_feedforward_enc,
                n_head=n_head_enc,
                dropout=dropout_enc)
        self.rul_Transformer=RUL_Transformer(input_dim=input_dim_rul,
                embed_dim=embed_dim_rul,
                head_dim=head_dim_rul,
                dim_feedforward=dim_feedforward_rul,
                n_head=n_head_rul,
                dropout=dropout_rul,
                mlp_dim=mlp_dim_rul,
                u_enc_dim=u_enc_dim_rul,#
                perform_w=pred_window_enc,  ###  
                dropout_pred=dropout_pred_rul,
                pred_window=pred_window_rul,
                )
        
    def forward(self, x,z,u):
        x_joint=torch.concatenate((x,z),axis=2)
        perform=self.transformer_enc(x_joint)     
        rul=self.rul_Transformer(perform,u)
        return rul



import math

class MLP(nn.Module):
    def __init__(self, 
                in_dim=[30,9],
                out_dim=[10], 
                #dim=[300,250,200,150,100,50]
                layers=[300,200,100]
                ):
        super().__init__()
        self.out_dim=out_dim
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
            nn.Linear(layers[2],math.prod(out_dim)),
        )    

    def forward(self, x):
        return self.encoder(x).squeeze(1)
    

    
class AutoencoderMLP(nn.Module):
    def __init__(self, 
                in_dim=[3],
                layers=[100,200,300],
                emb_dim=[2]
                ):
        super().__init__()
        self.encoder = MLP(in_dim=in_dim,layers=layers[::-1],out_dim=emb_dim)
        self.decoder = MLP(in_dim=emb_dim,layers=layers,out_dim=in_dim)
                                        
    def forward(self, x):
        z=self.encoder(x)
        return self.decoder(z)