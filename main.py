import torch
from torch import Tensor


class Diffusor:

    def __init__(self, beta: float = 0.001, num_diff_steps: int = 100):
        self.beta = beta  # must be small, i.e. 0.001
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
        # stack batch_dim copies of beta_schedule
        #  to match the shape of x
        alpha_schedule: Tensor = (
            torch.ones(self.beta_schedule.shape) - self.beta_schedule
        )
        alpha_bar = torch.prod(alpha_schedule[:num_steps], dim=0)
        mu_scale: torch.Tensor = torch.sqrt(alpha_bar).repeat(x.shape)
        sigma_scale: torch.Tensor = 1 - alpha_bar.repeat(x.shape)

        # beta_vec: Tensor = torch.pow(self.beta * torch.ones(x.shape), num_steps)
        mean: Tensor = mu_scale * x
        sdev: Tensor = sigma_scale
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

    def __init__(self, input_dim: int, time_dim, output_dim: int):
        super(MLP, self).__init__()
        self.encoder = torch.nn.Linear(time_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc1 = torch.nn.Linear(input_dim + 1, 128)
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
        # print(x.shape, time.shape)
        time_enc = self.sigmoid(self.encoder(time))
        x = self.fc1(torch.cat([x, time_enc], dim=1))
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

    time_dim: int = 100  # one-hot-encoding of time steps
    sample_dim: int = 1
    output_dim: int = 2  # mu, sigma

    diffusor: Diffusor = Diffusor(0.99, time_dim)
    diff_mlp = MLP(input_dim=sample_dim, time_dim=time_dim, output_dim=output_dim)
    optimizer: torch.optim.Optimizer = torch.optim.SGD(diff_mlp.parameters())
    train(diffusor, diff_mlp, optimizer, gen_data(), num_train_steps=1000)


def gen_data() -> torch.Tensor:
    """Bimodal normal distribution."""
    n1: torch.distributions.Distribution = torch.distributions.Normal(-5, 1)
    n2: torch.distributions.Distribution = torch.distributions.Normal(5, 0.5)
    samples: torch.Tensor = torch.cat([n1.sample((10000,)), n2.sample((10000,))])
    return samples


def generate_samples(denoiser: MLP, num_samples: int) -> torch.Tensor:
    """Generate samples from the denoiser model."""
    time_steps: int = (
        denoiser.fc1.in_features - 1
    )  # assuming input_dim = time_dim + sample_dim
    noise: Tensor = torch.randn((num_samples,))  # +1 for sample_dim

    for denoise_step in range(time_steps - 1, 0, -1):
        time_vec: Tensor = torch.zeros((num_samples, time_steps))
        time_vec[:, denoise_step] = 1.0
        noise_param = denoiser(noise.unsqueeze(1), time_vec)
        mu_out = noise_param[:, 0]
        sigma_out = noise_param[:, 1]
        noise = mu_out + sigma_out * torch.randn_like(mu_out)

    return noise


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

    losses = []
    for train_step in range(num_train_steps):

        # get batch and time step
        batch_idcs: torch.Tensor = torch.randint(0, data_0.shape[0], (batch_size,))
        batch: torch.Tensor = data_0[batch_idcs]
        time_step: int = torch.randint(1, diff_steps - 1, size=(1,)).item()
        data_t: Tensor = diffusor(batch, time_step)
        time_vec = torch.zeros((batch.shape[0], diff_steps))

        if time_step == 1:
            time_vec[:, 0] = 1.0
            out = denoiser(data_t.unsqueeze(1), time_vec)
            mu_out = out[:, 0]
            sigma_out = out[:, 1]
            likelihood = torch.distributions.Normal(mu_out, sigma_out).log_prob(batch)
            loss = -torch.mean(likelihood)

        else:
            time_vec[:, time_step] = 1.0
            # t is in the middle of the diffusion process --> consistency loss
            data_t = diffusor.n_step(batch, num_steps=time_step)

            # parameters of q(x_{t-1} | x_t, x_0)
            q_mu = (
                torch.sqrt(torch.prod(alpha_schedule[: time_step - 1]))
                * beta_schedule[time_step]
                * batch
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

            # parameters for p(x_{t-1} | x_t)
            out = denoiser(data_t.unsqueeze(1), time_vec)

            # The following slightly diverges from the source (struemke23)
            p_mu = out[:, 0]  # mu_t
            p_sigma = out[:, 1]  # sigma_t

            optimizer.zero_grad()
            loss = kl_div_different_normal(q_mu, p_mu, q_sigma, p_sigma).mean()

        assert not loss.isnan(), "Loss is NaN"
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    return losses


if __name__ == "__main__":
    main()
