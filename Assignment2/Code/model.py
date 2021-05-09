import torch
import torch.nn as nn

from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron
import torch.nn.functional as F


class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, num_feature_columns):
        super().__init__()
        self.field_dims = field_dims
        linear_in = len(field_dims) + num_feature_columns
        self.linear = FeaturesLinear(field_dims, linear_in)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = (len(field_dims) * embed_dim) + num_feature_columns
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear_out = nn.Linear(in_features=6, out_features=3) # 1 for linear, 1 for FM and 4 for MLP

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x.squeeze(1)
        x_cat = x[:,:len(self.field_dims)].long()
        x_feat = x[:,len(self.field_dims):]


        embed_x = self.embedding(x_cat)
        linear = self.linear(x_cat, x_feat)
        x_emb = torch.cat([embed_x, x_feat], dim=1)
        FM = self.fm(x_emb)
        MLP = self.mlp(x_emb.view(-1, self.embed_output_dim))
        x = torch.cat([linear, FM, MLP], axis=1)
        # out = F.softmax(self.linear_out(x), dim=0)
        out = self.linear_out(x)
        return out
