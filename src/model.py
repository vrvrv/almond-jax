from typing import Sequence

from flax import linen as nn
import ml_collections


class MLP_Sigmoid(nn.Module):
    hidden_dims: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(dim) for dim in self.hidden_dims]

    def __call__(self, z):
        for i, lyr in enumerate(self.layers):
            z = nn.tanh(lyr(z))

        return nn.sigmoid(z)


def get_model(model_config: ml_collections.ConfigDict):
    if model_config.architecture == 'mlp':
        return MLP_Sigmoid(model_config.hidden_dims)
