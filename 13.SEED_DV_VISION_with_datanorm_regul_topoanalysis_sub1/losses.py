import torch
import torch.nn.functional as F

def batch_hard_triplet_loss(embeddings, labels, margin=1.0, squared=False):
    """
    Compute batch-hard triplet loss.
    
    Args:
        embeddings (torch.Tensor): Batch of embeddings, shape (batch_size, embedding_dim)
        labels (torch.Tensor): Labels for the embeddings, shape (batch_size,)
        margin (float): Margin for triplet loss
        squared (bool): If True, use squared Euclidean distance; else, use Euclidean distance
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Compute pairwise distance matrix (batch_size x batch_size)
    pairwise_dist = _pairwise_distance(embeddings, squared=squared)
    
    # Initialize loss
    batch_size = embeddings.size(0)
    losses = []
    
    for i in range(batch_size):
        anchor_label = labels[i]
        
        # Create mask for positive and negative samples
        positive_mask = (labels == anchor_label) & (torch.arange(batch_size).to(labels.device) != i)
        negative_mask = (labels != anchor_label)
        
        # Get distances for positives and negatives
        positive_dists = pairwise_dist[i][positive_mask]
        negative_dists = pairwise_dist[i][negative_mask]
        
        if positive_dists.size(0) == 0 or negative_dists.size(0) == 0:
            # Skip if no valid positives or negatives
            continue
        
        # Select hardest positive (maximum distance)
        hardest_positive_dist = positive_dists.max()
        
        # Select hardest negative (minimum distance)
        hardest_negative_dist = negative_dists.min()
        
        # Compute triplet loss for this anchor
        loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
        losses.append(loss)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    
    # Average the losses
    return torch.stack(losses).mean()

def _pairwise_distance(embeddings, squared=False):
    """
    Compute pairwise distance matrix.
    
    Args:
        embeddings (torch.Tensor): Batch of embeddings, shape (batch_size, embedding_dim)
        squared (bool): If True, return squared Euclidean distance
    
    Returns:
        torch.Tensor: Pairwise distance matrix, shape (batch_size, batch_size)
    """
    # Compute dot product
    dot_product = torch.matmul(embeddings, embeddings.t())
    
    # Compute squared norm of each embedding
    square_norm = torch.diag(dot_product)
    
    # Compute pairwise distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    distances = square_norm.unsqueeze(0) + square_norm.unsqueeze(1) - 2.0 * dot_product
    
    # Ensure numerical stability
    distances = torch.clamp(distances, min=0.0)
    
    if not squared:
        # Apply square root to get Euclidean distance
        distances = torch.sqrt(distances + 1e-16)
    
    return distances