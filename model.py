import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DeepGene_AIID(nn.Module):
    def __init__(self, expr_dim, mut_dim, hidden_dim, out_dim=1, dropout=0.3):
        """
        DeepGene-AIID: A Multi-modal Dual-channel Graph Neural Network
        
        Args:
            expr_dim (int): Dimension of transcription/expression features
            mut_dim (int): Dimension of mutation features
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output dimension for prediction (default=1 for probability)
            dropout (float): Dropout probability
        """
        super(DeepGene_AIID, self).__init__()
        self.dropout = dropout
        
        # ==========================================
        # Module 1: Independent Dual-channel Feature Extraction
        # Expression and mutation data enter Channel A and B separately
        # ==========================================
        # Channel A: Extracts transcription/expression features
        self.channel_A = GCNConv(expr_dim, hidden_dim)
        
        # Channel B: Extracts mutation features
        self.channel_B = GCNConv(mut_dim, hidden_dim)
        
        # ==========================================
        # Module 2: Feature Fusion Layer
        # ==========================================
        # Fusion of outputs from Channel A and Channel B
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )
        
        # ==========================================
        # Module 3: Feature Disentanglement Layer
        # Disentangles the fused features into mutation, transcription, and structural features
        # ==========================================
        self.disentangle_mut = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.disentangle_trans = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.disentangle_struct = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # ==========================================
        # Module 4: Downstream Prediction Module
        # ==========================================
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()  # Outputs the probability of being a disease-associated gene
        )

    def forward(self, expr_x, mut_x, edge_index):
        # 1. Independent Inputs
        # expr_x enters Channel A, mut_x enters Channel B
        out_A = F.relu(self.channel_A(expr_x, edge_index))
        out_B = F.relu(self.channel_B(mut_x, edge_index))
        
        out_A = F.dropout(out_A, p=self.dropout, training=self.training)
        out_B = F.dropout(out_B, p=self.dropout, training=self.training)
        
        # 2. Feature Fusion
        fused_features = torch.cat([out_A, out_B], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # 3. Feature Disentanglement
        z_mut = self.disentangle_mut(fused_features)
        z_trans = self.disentangle_trans(fused_features)
        z_struct = self.disentangle_struct(fused_features)
        
        # 4. Prediction Scoring
        # Concatenate the three disentangled features for final prediction
        disentangled_concat = torch.cat([z_mut, z_trans, z_struct], dim=1)
        pred_score = self.predictor(disentangled_concat)
        
        return pred_score, z_mut, z_trans, z_struct