import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepGeneLoss(nn.Module):
    def __init__(self, alpha=0.1):
        """
        Combined loss function for DeepGene-AIID
        
        Args:
            alpha (float): Weight for the disentanglement orthogonal penalty term
        """
        super(DeepGeneLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()

    def orthogonal_loss(self, z1, z2):
        """
        Calculates the orthogonal constraint loss between two feature matrices 
        to enforce independence in the latent space.
        """
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        corr = torch.matmul(z1.T, z2)  # Cosine similarity matrix
        identity = torch.eye(corr.size(0)).to(corr.device)
        # Penalize non-diagonal elements of the similarity matrix
        return torch.norm(corr - identity, p='fro')

    def forward(self, pred_score, labels, z_mut, z_trans, z_struct):
        # 1. Main Prediction Loss (BCE)
        task_loss = self.bce_loss(pred_score.squeeze(), labels.float())
        
        # 2. Disentanglement Constraint Loss 
        # Ensures mutation, transcription, and structural features are mutually independent
        loss_orth_1 = self.orthogonal_loss(z_mut, z_trans)
        loss_orth_2 = self.orthogonal_loss(z_mut, z_struct)
        loss_orth_3 = self.orthogonal_loss(z_trans, z_struct)
        
        disentangle_loss = loss_orth_1 + loss_orth_2 + loss_orth_3
        
        # 3. Total Loss computation
        total_loss = task_loss + self.alpha * disentangle_loss
        
        return total_loss, task_loss, disentangle_loss