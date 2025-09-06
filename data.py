import torch


def gen_data(len: int = 1000) -> torch.Tensor:
    """Bimodal normal distribution."""
    n1: torch.distributions.Distribution = torch.distributions.Normal(-10, 1)
    n2: torch.distributions.Distribution = torch.distributions.Normal(5, 0.5)
    n3: torch.distributions.Distribution = torch.distributions.Normal(2, 0.8)
    samples: torch.Tensor = torch.cat(
        [n1.sample((len // 3,)), n2.sample((len // 3,)), n3.sample((len // 3,))]
    )
    return samples
