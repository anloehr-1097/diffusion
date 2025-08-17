import torch
from torch import Tensor


class Diffusor:

    def __init__(self, beta: float, num_diff_steps: int = 100):
        self.beta = beta
        self.beta_schedule = torch.ones(num_diff_steps) * beta
        self.diff_steps = num_diff_steps

    def __call__(self, x: Tensor, steps: int = 100) -> float:
        """
        Apply the diffusor to the input value x.

        :param x: The input value to be processed by the diffusor.
        :return: The processed value after applying the diffusor.
        """
        for _ in range(steps):
            x = self.step(x)

        return x

    def n_step(self, x: Tensor, num_steps: int = 100) -> Tensor:
        """
        Apply the diffusor to the input value x for a specified number of steps.

        :param x: The input value to be processed by the diffusor.
        :param num_steps: The number of steps to apply the diffusor.
        :return: The processed value after applying the diffusor for num_steps.
        """
        beta_vec: Tensor = torch.pow(self.beta * torch.ones(x.shape), num_steps)
        mean: Tensor = torch.sqrt(torch.ones(beta_vec.shape) - beta_vec) * x
        sdev: Tensor = torch.sqrt(beta_vec)
        return torch.normal(mean, sdev)

    def step(self, x: Tensor) -> float:
        """
        Perform a single step of the diffusion process.

        :param x: The input value to be processed in this step.
        :return: The processed value after one step of diffusion.
        """
        beta_vec: Tensor = torch.ones(x.shape) * self.beta
        mean: Tensor = torch.sqrt(torch.ones(beta_vec.shape) - beta_vec) * x
        sdev: Tensor = torch.sqrt(beta_vec)
        return torch.normal(mean, sdev)


class MLP(torch.nn.Module):
    """Parameterize Denoising Process with MLP.

    As a map, this is (x_t, t) -> (mu_t, sigma_t)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, output_dim)
        self.relu = torch.nn.ReLU()

    def __call__(self, x: Tensor, time: Tensor) -> Tensor:
        """
        Forward pass through the MLP.

        :param x: Input tensor.
        :param time: Time tensor for conditioning.
        :return: Output tensor after passing through the MLP.
        """
        time = time / 100
        print(x.shape, time.shape)
        x = self.fc1(torch.cat([x, time], dim=1))
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # Ensure output is non-negative for sigma
        x[:, 1] = torch.exp(x[:, 1])
        return x


def loss():
    """Loss used for diffusion training.


    Lower bound to negative log likelihood of the data.

    Terms:
        - reconstruction term = E_{q(x_1 | x_0)}[log p(x_0 | x_1)]
        - (Prior matching) - disregarded as no parameter dependence
        - consistency terms  = E_{q(x_t | x_0)}[D_KL(q(x_{t-1} x_t, x_0) || p(x_{t-1} | x_t))]
    """
    pass


def kl_div_normal(mu1: Tensor, mu2: Tensor, sigma: Tensor) -> Tensor:
    """Calculate KL divergence between two normal distributions.

    Assume sigma is the same for both distributions and sigma is
    s.t. variance = sigma^2 * I.

    Calc according to (27) in struemke23.
    """
    return torch.sum(torch.square(mu1 - mu2)) * 1 / 2 * sigma


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


def main() -> None:

    time_dim: int = 1
    sample_dim: int = 1
    output_dim: int = 2  # mu, sigma

    diffusor: Diffusor = Diffusor(0.99)
    diff_mlp = MLP(input_dim=time_dim + sample_dim, output_dim=2)
    optimizer: torch.optim.Optimizer = torch.optim.SGD(diff_mlp.parameters())
    train(diffusor, diff_mlp, optimizer, gen_data(), num_train_steps=1000)


def gen_data() -> torch.Tensor:
    """Bimodal normal distribution."""
    n1: torch.distributions.Distribution = torch.distributions.Normal(-5, 1)
    n2: torch.distributions.Distribution = torch.distributions.Normal(5, 0.5)
    samples: torch.Tensor = torch.cat([n1.sample((5000,)), n2.sample((5000,))])
    return samples


def train(
    diffusor: Diffusor,
    denoiser: MLP,
    optimizer: torch.optim.Optimizer,
    data_0: torch.Tensor,
    num_train_steps: int = 1000,
    batch_size: int = 64,
):
    """Training diffusor model."""
    # sample a time step t uniformly
    diff_steps: int = diffusor.diff_steps
    beta_schedule: torch.Tensor = diffusor.beta_schedule
    alpha_schedule: Tensor = torch.ones(beta_schedule.shape[0]) - beta_schedule

    for train_step in range(num_train_steps):

        # get batch and time step
        batch_idcs: torch.Tensor = torch.randint(0, data_0.shape[0], (batch_size,))
        batch: torch.Tensor = data_0[batch_idcs]
        time_step: int = torch.randint(1, diff_steps - 1, size=(1,)).item()
        data_t: Tensor = diffusor(batch, time_step)

        if time_step == 1:
            out = denoiser(
                data_t.unsqueeze(1),
                torch.ones(data_t.shape[0]).unsqueeze(1) * time_step,
            )
            mu_out = out[:, 0]
            sigma_out = out[:, 1]
            likelihood = torch.distributions.Normal(mu_out, sigma_out).log_prob(batch)
            loss = -torch.mean(likelihood)

        else:
            # t is in the middle of the diffusion process --> consistency loss
            data_t = diffusor.n_step(data_0, num_steps=time_step)
            q_mu = (
                torch.sqrt(torch.prod(alpha_schedule[: time_step - 1]))
                * beta_schedule[time_step]
                * data_0
                / (1 - torch.prod(alpha_schedule[:time_step]))
            ) + torch.sqrt(alpha_schedule[time_step]) * (
                1 - torch.prod(alpha_schedule[: time_step - 1])
            ) * data_t / (
                1 - torch.prod(alpha_schedule[:time_step])
            )

            q_sigma = (
                (1 - torch.prod(alpha_schedule[: time_step - 1]))
                * beta_schedule[time_step]
                / (1 - torch.prod(alpha_schedule[:time_step]))
            )

            out = denoiser(
                data_t.unsqueeze(1),
                torch.ones(data_t.shape[0]).unsqueeze(1) * time_step,
            )

            # The following slightly diverges from the source (struemke23)
            p_mu = out[:, 0]  # mu_t
            p_sigma = out[:, 1]  # sigma_t

            optimizer.zero_grad()
            loss = kl_div_different_normal(q_mu, p_mu, q_sigma, p_sigma).mean()

        assert not loss.isnan(), "Loss is NaN"
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
