import torch
import torch.nn as nn
from .encoders import DualChannelEncoder
from .disentanglement import AdversarialDisentangler
from .simam import SimAMFusion


class DeepGeneAIID(nn.Module):
    """
    End-to-end DeepGene-AIID framework:
      1. Modality-specific Dual-Channel Graph Encoding (Weighted GraphSAGE + DiGCN)
      2. Adversarial Representation Disentanglement (Shared + 3 Modality-Specific)
      3. SimAM-Guided Feature Fusion
      4. MLP Classifier for AIID-Associated Gene Prioritization
    """

    def __init__(
        self,
        gen_dim: int,
        exp_dim: int,
        stru_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.3,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.gen_encoder = DualChannelEncoder(gen_dim, hidden_dim, hidden_dim, dropout)
        self.exp_encoder = DualChannelEncoder(exp_dim, hidden_dim, hidden_dim, dropout)
        self.stru_encoder = DualChannelEncoder(stru_dim, hidden_dim, hidden_dim, dropout)

        self.disentangler = AdversarialDisentangler(
            in_dim=hidden_dim,
            latent_dim=latent_dim,
            num_modalities=3,
            grl_lambda=grl_lambda,
        )
        self.simam_fusion = SimAMFusion(e_lambda=1e-4)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_gen: torch.Tensor,
        x_exp: torch.Tensor,
        x_stru: torch.Tensor,
        ppi_edge_index: torch.Tensor,
        dir_edge_index: torch.Tensor,
        ppi_edge_weight: torch.Tensor = None,
        dir_edge_weight: torch.Tensor = None,
    ):
        h_gen = self.gen_encoder(x_gen, ppi_edge_index, dir_edge_index, ppi_edge_weight, dir_edge_weight)
        h_exp = self.exp_encoder(x_exp, ppi_edge_index, dir_edge_index, ppi_edge_weight, dir_edge_weight)
        h_stru = self.stru_encoder(x_stru, ppi_edge_index, dir_edge_index, ppi_edge_weight, dir_edge_weight)

        z_shared, z_specs, aux_losses = self.disentangler([h_gen, h_exp, h_stru])
        fused = self.simam_fusion([z_shared, z_specs[0], z_specs[1], z_specs[2]])
        logits = self.classifier(fused).squeeze(-1)

        return logits, {
            "z_shared": z_shared,
            "z_gen": z_specs[0],
            "z_exp": z_specs[1],
            "z_stru": z_specs[2],
            **aux_losses,
        }
