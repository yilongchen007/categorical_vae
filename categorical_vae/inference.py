import torch
from .utils import categorical_kl_divergence


def evaluate_model(model, test_dataloader, device, temperature):
    """
    Evaluates the model on the test dataset and computes average reconstruction and KL losses.

    Parameters:
    - model (torch.nn.Module): The trained model to evaluate.
    - test_dataloader (DataLoader): DataLoader for the test set.
    - device (torch.device): The device (CPU or GPU) to run the evaluation on.
    - temperature (float): Temperature parameter for the model (used in forward pass if applicable).

    Returns:
    - average_reconstruction_loss (float): Average reconstruction loss over the test set.
    - average_kl_loss (float): Average KL divergence loss over the test set.
    - average_total_loss (float): Average total loss (reconstruction + KL) over the test set.
    """
    
    # Initialize accumulators
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient computation for evaluation
        for data in test_dataloader:
            # Move data to the appropriate device
            x = data.to(device)
            
            # Forward pass
            phi, x_prob = model(x, temperature)
            
            # Compute reconstruction loss
            reconstruction_loss = sum([
                torch.nn.functional.cross_entropy(x_prob[i], x[:, i]) for i in range(x.shape[1])
            ]) / x.shape[0]
            
            # Compute KL divergence loss
            kl_loss = torch.mean(torch.sum(categorical_kl_divergence(phi), dim=1))
            
            # Update accumulators
            total_reconstruction_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

    # Calculate average losses over the test set
    average_reconstruction_loss = total_reconstruction_loss / num_batches
    average_kl_loss = total_kl_loss / num_batches
    average_total_loss = average_reconstruction_loss + average_kl_loss

    # Print results
    print(f"Average Reconstruction Loss: {average_reconstruction_loss}")
    print(f"Average KL Loss: {average_kl_loss}")
    print(f"Average Total Loss: {average_total_loss}")

    return average_reconstruction_loss, average_kl_loss, average_total_loss

