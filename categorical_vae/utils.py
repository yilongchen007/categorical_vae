import torch
import torch.distributions as dist
import numpy as np
import argparse


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cpu')

    # Data & Saving
    parser.add_argument('--data_path', type=str, default='./data/data_compressed.npz')
    parser.add_argument('--save_path', type=str, default='./cache')
    parser.add_argument('--version', type=str, default='v0')

    # Model config
    parser.add_argument('--encoder_type', type=str, default='transformer')
    parser.add_argument('--decoder_type', type=str, default='simple')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--K', type=int, default=64)

    # Training config
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--initial_lr', type=float, default=0.05)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--max_steps', type=int, default=25000)
    parser.add_argument('--model_save_interval', type=int, default=5000)

    # Temperature config
    parser.add_argument('--initial_temp', type=float, default=1.0)
    parser.add_argument('--min_temp', type=float, default=0.01)
    parser.add_argument('--temp_decay', type=float, default=0.0001)

    # Optional CP initialization
    parser.add_argument('--cp_path', type=str, default='', help='Path to CP factor .pkl file')

    args = parser.parse_args()
    return vars(args)


def split_tensor_efficient(tensor, train_ratio=0.8, seed=42):
    np.random.seed(seed)  # Ensure reproducibility
    
    # Get the indices of all nonzero values
    nonzero_indices = np.nonzero(tensor)  # Tuple of arrays
    
    # Flattened indices for random selection
    num_nonzero = len(nonzero_indices[0])
    train_size = int(num_nonzero * train_ratio)
    
    # Randomly select indices for the training set
    train_mask = np.zeros(num_nonzero, dtype=bool)
    train_mask[np.random.choice(num_nonzero, size=train_size, replace=False)] = True
    
    # Initialize empty tensors
    train_tensor = np.zeros_like(tensor)
    test_tensor = np.zeros_like(tensor)
    
    # Assign values based on the random mask
    train_tensor[nonzero_indices] = tensor[nonzero_indices] * train_mask
    test_tensor[nonzero_indices] = tensor[nonzero_indices] * (~train_mask)

    return train_tensor, test_tensor


def gumbel_distribution_sample(shape: torch.Size, eps=1e-20) -> torch.Tensor:
    """Samples from the Gumbel distribution given a tensor shape and value of epsilon.
    
    note: the \eps here is just for numerical stability. The code is basically just doing
            > -log(-log(rand(shape)))
    where rand generates random numbers on U(0, 1). 
    """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_distribution_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Adds Gumbel noise to `logits` and applies softmax along the last dimension.
    
    Softmax is applied wrt a given temperature value. A higher temperature will make the softmax
    softer (less spiky). Lower temperature will make softmax more spiky and less soft. As
    temperature -> 0, this distribution approaches a categorical distribution.
    """
    assert len(logits.shape) == 2 # (should be of shape (b, n_classes))
    y = logits + gumbel_distribution_sample(logits.shape).to(logits.device)
    return torch.nn.functional.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits: torch.Tensor, temperature: float, batch=False) -> torch.Tensor:
    """
    Gumbel-softmax.
    input: [*, n_classes] (or [b, *, n_classes] for batch)
    return: flatten --> [*, n_class] a one-hot vector (or b, *, n_classes for batch)
    """
    input_shape = logits.shape
    if batch:
        assert len(logits.shape) == 3
        b, n, k = input_shape
        logits = logits.view(b*n, k)
    assert len(logits.shape) == 2
    y = gumbel_softmax_distribution_sample(logits, temperature)    
    return y.view(input_shape)


def categorical_kl_divergence(phi: torch.Tensor) -> torch.Tensor:
    # phi is logits of shape [B, N, K] where B is batch, N is number of categorical distributions, K is number of classes
    B, N, K = phi.shape
    phi = phi.view(B*N, K)
    q = dist.Categorical(logits=phi)
    p = dist.Categorical(probs=torch.full((B*N, K), 1.0/K, device=phi.device)) # uniform bunch of K-class categorical distributions
    kl = dist.kl.kl_divergence(q, p) # kl is of shape [B*N]
    return kl.view(B, N)

