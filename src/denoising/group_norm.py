import torch


def group_norm(x, gamma, beta, G, eps=1e-5):
    """
    Group normalization introduced by Yuxin Wu and Kaiming He in the paper 'Group normalization', 2018
    The original code was using TensorFlow, we translated it to pytorch.

    Here N is the batch axis, C is the channel axis, and H and W are the spatial height and width axes.
    """
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    N, C, H, W = x.shape
    x = torch.reshape(x, [N, G, C // G, H, W])
    var, mean = torch.var_mean(x, dim=(2, 3, 4), keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    x = torch.reshape(x, [N, C, H, W])
    return x * gamma + beta
