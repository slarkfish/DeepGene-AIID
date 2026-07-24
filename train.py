import torch
from model import DeepGene_AIID
from loss import DeepGeneLoss

def train_deepgene_aiid(expr_data, mut_data, edge_index, labels, epochs=200, lr=0.005):
    """
    Training pipeline for DeepGene-AIID
    
    Args:
        expr_data (Tensor): Transcription/Expression node features
        mut_data (Tensor): Mutation node features
        edge_index (Tensor): Graph connectivity (edge index)
        labels (Tensor): Ground truth labels for nodes
        epochs (int): Number of training epochs
        lr (float): Learning rate
        
    Returns:
        model (nn.Module): The trained DeepGene-AIID model
    """
    # Detect available hardware configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training started on device: {device}")
    
    # Initialize the model architecture
    model = DeepGene_AIID(
        expr_dim=expr_data.shape[1], 
        mut_dim=mut_data.shape[1], 
        hidden_dim=128
    ).to(device)
    
    # Initialize the customized loss criterion and optimizer
    criterion = DeepGeneLoss(alpha=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    # Transfer data to the specified device
    expr_data = expr_data.to(device)
    mut_data = mut_data.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)
    
    # Start training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass (extracts prediction scores and disentangled features)
        pred_scores, z_mut, z_trans, z_struct = model(expr_data, mut_data, edge_index)
        
        # Calculate combined losses
        loss, task_loss, dis_loss = criterion(pred_scores, labels, z_mut, z_trans, z_struct)
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Print training progress
        if (epoch + 1) % 20 == 0:
            print(f'Epoch: {epoch + 1:03d}/{epochs}, Total Loss: {loss.item():.4f}, '
                  f'BCE Loss: {task_loss.item():.4f}, Disentangle Loss: {dis_loss.item():.4f}')
            
    print("Training process completed successfully.")
    return model