import torch
from torch import Tensor


class Diffusor:

    def __init__(self, beta_max: float = 0.05, num_diff_steps: int = 1000):
        # self.beta = beta  # must be small, i.e. 0.001

        self.beta_schedule: Tensor = torch.linspace(1e-4, beta_max, num_diff_steps)
        self.alpha_schedule: Tensor = torch.ones(num_diff_steps) - self.beta_schedule

        self.diff_steps = num_diff_steps

    def n_step(self, x: Tensor, n: int = 100) -> Tensor:
        """
        Apply the diffusor to the input value x for a specified number of steps.

        :param x: The input value to be processed by the diffusor.
        :param num_steps: The number of steps to apply the diffusor.
        :return: The processed value after applying the diffusor for num_steps.
        """
        # stack batch_dim copies of beta_schedule
        #  to match the shape of x
        alpha_to_t = torch.prod(self.alpha_schedule[:n], dim=0)  # \overline{alpha}_t}
        mu_scale: torch.Tensor = torch.sqrt(alpha_to_t).repeat(x.shape)
        sigma_scale: torch.Tensor = (1 - alpha_to_t).repeat(x.shape)

        # beta_vec: Tensor = torch.pow(self.beta * torch.ones(x.shape), num_steps)
        mean: Tensor = mu_scale * x
        sdev: Tensor = sigma_scale

        return mean + sdev * torch.randn_like(mean)
