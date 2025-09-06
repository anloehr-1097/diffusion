import torch
from torch import Tensor


def kl_div_normal(mu1: Tensor, mu2: Tensor, sigma: Tensor) -> Tensor:
    """Calculate KL divergence between two normal distributions.

    Assume sigma is the same for both distributions and sigma is
    s.t. variance = sigma^2 * I.

    Calc according to (27) in struemke23.
    """

    loss = torch.sum(torch.square(mu1 - mu2)) / (2 * sigma)
    return loss


def kl_div_different_normal(
    mu1: Tensor, mu2: Tensor, sigma1: Tensor, sigma2: Tensor
) -> Tensor:
    """Calculate KL divergence between two normal distributions.

    Assume sigma is the same for both distributions and sigma is
    s.t. variance = sigma^2 * I.

    Calc according to (27) in struemke23.
    """
    return (
        1
        / 2
        * (
            torch.log(sigma2 / sigma1)
            - 1
            + (sigma1 / sigma2)
            + 1 / sigma2 * torch.sum(torch.square(mu1 - mu2))
        )
    )
    # return torch.sum(torch.square(mu1 - mu2)) * 1 / 2 * sigma
