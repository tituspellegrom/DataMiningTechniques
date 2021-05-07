import torch
import torch.nn as nn

from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear_out = nn.Linear(in_features=9, out_features=3)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # TODO split embeddings and features
        embed_x = self.embedding(x)
        linear = self.linear(x)
        FM = self.fm(embed_x)
        MLP = self.mlp(embed_x.view(-1, self.embed_output_dim))
        x = torch.cat([linear, FM, MLP], axis=1)
        out = self.linear_out(x)
        return x
