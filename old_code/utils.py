from torch import nn
import torch


def fc(layer_config, sparsity=-1, activation=nn.ELU, bias=False):
    layers = []
    for i in range(len(layer_config) - 1):
        lin = nn.Linear(*layer_config[i:i+2], bias=bias)
        if sparsity > 0 and i < len(layer_config) - 2:
            k = (sparsity - 1) / 2
            mask = torch.tensor([
                [int(abs(i-j) <= k) for i in range(layer_config[i])]
                for j in range(layer_config[i+1])],
                dtype=torch.float)
            lin.weight.data *= mask
        layers.append(lin)
        if i < len(layer_config) - 2:
            layers.append(activation())

    return nn.Sequential(*layers)


def weights_from_pytorch(module):
    r"""
    Extract weights and biases of the linear layers of a network
    """
    weights = []
    biases = []
    for layer in module:
        if type(layer) is nn.Linear or 'MaskedLinear' in str(type(layer)):
            weight = layer.weight
            bias = layer.bias
            try:
                weight = weight.detach().numpy()
            except Exception:
                pass
            try:
                bias = bias.detach().numpy()
            except Exception:
                pass
            weights.append(weight)
            biases.append(bias)

    return weights, biases

