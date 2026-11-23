import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer (GRL) for adversarial modality confusion."""

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor):
        return _GradReverse.apply(x, self.lambd)


class AdversarialDisentangler(nn.Module):
    """
    Disentangles encoded multimodal features into:
      - Shared cross-modal consensus representation (z_shared)
      - Modality-specific representations (z_gen, z_exp, z_stru)
    Uses an adversarial modality discriminator (with GRL) and orthogonality constraints.
    """

    def __init__(self, in_dim: int, latent_dim: int, num_modalities: int = 3, grl_lambda: float = 1.0):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.specific_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
            )
            for _ in range(num_modalities)
        ])
        self.grl = GradientReversalLayer(lambd=grl_lambda)
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_modalities),
        )

    @staticmethod
    def orthogonality_loss(z_shared: torch.Tensor, z_spec: torch.Tensor) -> torch.Tensor:
        z_s = F.normalize(z_shared, p=2, dim=-1)
        z_p = F.normalize(z_spec, p=2, dim=-1)
        return torch.mean((z_s * z_p).sum(dim=-1) ** 2)

    def forward(self, h_list):
        """
        Args:
            h_list: List of 3 tensors [h_gen, h_exp, h_stru], each of shape (N, in_dim).
        Returns:
            z_shared_consensus: (N, latent_dim)
            z_specific_list: List of 3 tensors [z_gen, z_exp, z_stru], each (N, latent_dim)
            aux_losses: Dict containing adversarial loss and orthogonality loss
        """
        num_nodes = h_list[0].size(0)
        device = h_list[0].device

        z_shared_per_mod = [self.shared_encoder(h) for h in h_list]
        z_specific_list = [enc(h) for enc, h in zip(self.specific_encoders, h_list)]

        # Consensus shared representation (mean across modalities)
        z_shared_consensus = torch.stack(z_shared_per_mod, dim=0).mean(dim=0)

        # Adversarial loss on shared representations (encourage modality invariance)
        adv_loss = 0.0
        ortho_loss = 0.0
        for mod_idx, (z_s, z_p) in enumerate(zip(z_shared_per_mod, z_specific_list)):
            mod_labels = torch.full((num_nodes,), mod_idx, dtype=torch.long, device=device)
            logits_s = self.discriminator(self.grl(z_s))
            adv_loss = adv_loss + F.cross_entropy(logits_s, mod_labels)
            ortho_loss = ortho_loss + self.orthogonality_loss(z_s, z_p)

        adv_loss = adv_loss / len(h_list)
        ortho_loss = ortho_loss / len(h_list)

        return z_shared_consensus, z_specific_list, {
            "adv_loss": adv_loss,
            "ortho_loss": ortho_loss,
        }
