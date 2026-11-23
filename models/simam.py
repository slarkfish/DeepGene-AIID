import torch
import torch.nn as nn


class SimAMFusion(nn.Module):
    """
    Parameter-free Simple Attention Module (SimAM) adapted for multimodal
    gene representation fusion across the 4 disentangled channels:
    [shared-consensus, genomic-specific, expression-specific, structure-specific].
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()

    def forward(self, reps: list) -> torch.Tensor:
        """
        Args:
            reps: List of 4 tensors [z_shared, z_gen, z_exp, z_stru], each of shape (N, D).
        Returns:
            fused: Tensor of shape (N, 4 * D) weighted by SimAM energy attention.
        """
        # Stack into (N, C=4, D)
        x = torch.stack(reps, dim=1)
        d = x.size(-1)

        mu = x.mean(dim=-1, keepdim=True)
        var = ((x - mu) ** 2).sum(dim=-1, keepdim=True) / max(d - 1, 1)
        energy = ((x - mu) ** 2) / (4.0 * (var + self.e_lambda)) + 0.5
        attn = self.activation(energy)

        weighted = x * attn
        return weighted.reshape(x.size(0), -1)
