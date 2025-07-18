# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, dropout=0.0):
        super(MLP, self).__init__()
        self.mlp_blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.in_dropout = nn.Dropout(p=dropout)
        self.out_projection = nn.Linear(hidden_sizes[-1], latent_size)
        hidden_sizes = [input_size] + hidden_sizes
        for idx, (input_size, output_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            self.mlp_blocks.append(
                nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.LayerNorm(output_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                )
            )
            # add residual connections
            self.residuals.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=output_size,
                    kernel_size=input_size,
                    bias=False,
                    stride=input_size,
                )
            )

    def forward(self, x):
        x = self.in_dropout(x)
        for i in range(len(self.mlp_blocks)):
            res = self.residuals[i](x.unsqueeze(1)).squeeze()
            x = self.mlp_blocks[i](x)
            x = x + res
        return self.out_projection(x)


class MLPBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(MLPBlock, self).__init__()
        self.ln = nn.LayerNorm(input_size, elementwise_affine=True)
        self.up = nn.Linear(input_size, hidden_size)
        self.down = nn.Linear(hidden_size, input_size)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        res = x.clone()
        x = self.dropout(self.down(self.act_fn(self.up(self.ln(x)))))
        return res + x


class QuantizationLayer(nn.Module):
    # codebook_size can be either an int (meaning we use same codebook_size for all levels),
    # or a list of int to specify the codebook_size for each codebook level.
    def __init__(self, num_levels, codebook_size, latent_size):
        super(QuantizationLayer, self).__init__()
        self.num_levels = num_levels
        self.latent_size = latent_size

        # Check if codebook_size is an int and convert it to a list of the same size for each level
        if isinstance(codebook_size, int):
            self.codebook_sizes = [codebook_size] * num_levels
        elif isinstance(codebook_size, list) and len(codebook_size) == num_levels:
            self.codebook_sizes = codebook_size
        else:
            raise ValueError(
                "codebook_size must be an int or a list of int with length equal to num_levels"
            )

        self.codebooks = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.codebook_sizes[l], latent_size))
                for l in range(num_levels)
            ]
        )

    def forward(self, x):
        batch_size, _ = x.shape
        output = torch.empty(
            batch_size, self.num_levels, dtype=torch.long, device=x.device
        )
        # x -> batch_size * 32
        # codebook -> codebook_size*32
        r = []
        e = []
        count = 0
        z_hat = x
        for l in range(self.num_levels):
            r.append(x)
            x = x.unsqueeze(1)  # (batch_size, 1, latent_size)
            distances = torch.norm(
                x - self.codebooks[l].unsqueeze(0), dim=-1
            )  # Calculate distances to codebook entries
            indices = torch.argmin(distances, dim=-1)
            quantization = (
                self.codebooks[l].unsqueeze(0)[:, indices, :].squeeze(0)
            )  # Perform quantization
            output[:, l] = indices
            x = x.squeeze(1)
            counts = torch.bincount(indices, minlength=self.codebook_sizes[l])
            # Find codebook entries with usage < 1.0
            low_usage_indices = torch.where(counts < 1.0)[0]
            count += len(low_usage_indices)
            x = x - quantization
            e.append(quantization)

        e = torch.stack(e, dim=1)
        r = torch.stack(r, dim=1)
        z_hat = z_hat + (e.detach().sum(dim=1) - z_hat.detach())
        return output, r, e, z_hat, count

    def generate_codebook(self, x, device, seed):
        for l in range(self.num_levels):
            kmeans = KMeans(
                n_clusters=self.codebook_sizes[l], n_init="auto", random_state=seed
            ).fit(x.detach().cpu().numpy())
            self.codebooks[l].data = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float, device=device
            )
            x = x.unsqueeze(1)  # (batch_size, 1, latent_size)
            distances = torch.norm(
                x - self.codebooks[l].unsqueeze(0), dim=-1
            )  # Calculate distances to codebook entries
            indices = torch.argmin(distances, dim=-1)
            quantization = (
                self.codebooks[l].unsqueeze(0)[:, indices, :].squeeze(0)
            )  # Perform quantization
            x = x.squeeze(1)
            x = x - quantization
