import torch
from torch import nn
from .utils import gumbel_softmax
from typing import Tuple


class Encoder(nn.Module):
    def __init__(self, category_dims, embedding_dims, latent_shape: torch.Size, hidden_dim: int, network):
        """
        Encoder class with embedding layers for categorical inputs.

        Parameters:
        - category_dims (list of int): The number of unique values for each categorical variable.
        - embedding_dims (list of int): The embedding dimension for each categorical variable.
        - latent_shape (torch.Size): The shape of the output (latent) tensor.
        - hidden_dim (int): The number of hidden units in the linear layer.
        """

        super().__init__()
        self.category_dims = category_dims
        self.embedding_dims = embedding_dims
        self.latent_shape = latent_shape
        self.hidden_dim = hidden_dim
        self.network = network

        self.num_heads = 8
        self.num_layers = 6


        assert len(self.latent_shape) == 2

        # Create an embedding layer for each categorical variable
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim) for cat_dim, emb_dim in zip(category_dims, embedding_dims)
        ])

        # Define a linear layer that processes concatenated embeddings
        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sum(self.embedding_dims), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_shape[0] * self.latent_shape[1]),  # Output layer to produce parameters of a distribution
        )

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=sum(embedding_dims), nhead=self.num_heads, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=self.num_layers
        )
        
        # Linear projection to latent space
        self.output_layer = nn.Linear(sum(embedding_dims), self.latent_shape[0] * self.latent_shape[1])



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to produce encoding `z` for input `x`.

        Parameters:
        - x (torch.Tensor): Input tensor of shape [B, N], where B is the batch size and N is the number of categorical variables.

        Returns:
        - torch.Tensor: Output tensor of shape [B, N, K], where N is the number of categorical variables, and K is the number of output classes per variable.
        """
        assert len(x.shape) == 2  # x should be of shape [B, N]

        # Apply embeddings to each categorical variable
        embedded = [embed(x[:, i]) for i, embed in enumerate(self.embeddings)]
        
        # Concatenate all embeddings into a single vector
        x = torch.cat(embedded, dim=-1)

        if self.network == 'transformer':

            # Transformer requires input of shape [B, N, E] (batch size, sequence length, embedding size)
            x = x.unsqueeze(1)  # Reshape to [B, 1, E]
            x = self.transformer(x)  # Pass through Transformer [B, 1, E]
            x = x.squeeze(1)  # Reshape back to [B, E]

            output = self.output_layer(x)

        elif self.network == 'linear':
            output = self.linear_layer(x)

        else: 
            raise Exception

        
        # Reshape to [B, N, K] where K is the number of output classes per variable
        return output.view(-1, self.latent_shape[0], self.latent_shape[1])




class Decoder(nn.Module):
    def __init__(self, category_dims, latent_shape: torch.Size, hidden_dim: int, network):
        """
        Decoder that reconstructs categorical variables from latent space.

        Parameters:
        - category_dims (list of int): The number of unique values for each categorical variable.
        - latent_shape (torch.Size): The shape of the input (latent) tensor.
        - hidden_dim (int): The number of hidden units in the decoder.
        """
        super().__init__()
        self.category_dims = category_dims
        self.latent_shape = latent_shape
        self.hidden_dim = hidden_dim
        self.network = network

        self.num_heads = 8
        self.num_layers = 6

        assert len(self.latent_shape) == 2
        
        # Define the decoder network
        self.input_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.latent_shape[0]*self.latent_shape[1], self.hidden_dim),
            nn.ReLU(),
        )

        # Transformer decoder
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=self.num_heads, dim_feedforward=hidden_dim * 2, batch_first=True),
            num_layers=self.num_layers
        )

        # Output layers for reconstructing each categorical variable
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, cat_dim) for cat_dim in self.category_dims
        ])

        # # Output layers for each categorical variable

        self.linear_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        # Define M probability matrices (one for each categorical variable x_m)
        self.probability_matrices = nn.ModuleList([
            nn.Linear(self.latent_shape[1]**self.latent_shape[0], cat_dim, bias=False) for cat_dim in self.category_dims
        ])


    def batch_outer_product(self, tensor):
        """
        Compute the outer product of N one-hot vectors for each batch.

        Parameters:
        - tensor (torch.Tensor): Shape [B, N, K], where B = batch size, N = number of categorical variables, K = number of categories.

        Returns:
        - torch.Tensor: Shape [B, K^N], where each row is the joint one-hot representation.
        """
        B, N, K = tensor.shape  # B = batch size, N = number of variables, K = categories
        
        # Start with the first dimension of the outer product
        result = tensor[:, 0, :]  # Shape [B, K]
        
        for i in range(1, N):
            # Expand the current result and the next dimension to compute outer product
            result = result.unsqueeze(-1) * tensor[:, i, :].unsqueeze(1)  # Shape [B, K^(i), K]
            result = result.reshape(B, -1)  # Flatten to [B, K^(i+1)]

        assert result.shape == torch.Size([B, K**N])
        
        return result
    
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to produce logits for each categorical variable.
        
        Parameters:
        - z (torch.Tensor): Latent representation from encoder [B, sum(embedding_dims)].
        
        Returns:
        - List of torch.Tensor: Each tensor contains logits for a categorical variable, shape [B, K_i].
        """
        # # Pass through the main decoder network
        # z = self.network(z)
        
        # # Produce logits for each categorical variable
        # logits = [layer(z) for layer in self.output_layers]

        if self.network == 'transformer':

            z = self.input_layer(z).unsqueeze(1)  # [B, 1, hidden_dim]
            # Decode using the Transformer
            z = self.transformer(z, z)  # [B, 1, hidden_dim]
            z = z.squeeze(1)  # [B, hidden_dim]
            logits = [layer(z) for layer in self.output_layers]

            
        elif self.network == 'linear':
            z = self.input_layer(z)
            z = self.linear_layer(z)
            logits = [layer(z) for layer in self.output_layers]

        elif self.network == 'simple':
            z = self.batch_outer_product(z)
            logits = [layer(z) for layer in self.probability_matrices]


        else: 
            raise Exception
        

        # Optionally, you can convert these to probabilities with softmax
        probabilities = [torch.softmax(logit, dim=-1) for logit in logits]
        
        return logits  # or `return probabilities` if you need probabilities directly
    



class CategoricalVAE(torch.nn.Module):
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    temperature: float
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.temperature = 1.0

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """VAE forward pass. Encoder produces phi, the parameters of a categorical distribution.
        Samples from categorical(phi) using gumbel softmax to produce a z. Passes z through encoder p(x|z)
        to get x_hat, a reconstruction of x.

        Returns:
            phi: parameters of categorical distribution that produced z
            x_hat: auto-encoder reconstruction of x
        """
        phi = self.encoder(x)
        phi.retain_grad()

        z_given_x = gumbel_softmax(phi, temperature, batch=True)
        z_given_x.retain_grad()

        self._debug = {'phi':phi, 'z_given_x':z_given_x}

        x_prob = self.decoder(z_given_x)

        return phi, x_prob



