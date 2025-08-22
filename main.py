import torch
from torch import Tensor
import matplotlib.pyplot as plt
from dvclive import Live
from PIL import Image
import io


class Diffusor:

    def __init__(self, beta_max: float = 0.05, num_diff_steps: int = 1000):
        # self.beta = beta  # must be small, i.e. 0.001

        self.beta_schedule: Tensor = torch.linspace(10e-4, beta_max, num_diff_steps)
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

    # def step(self, x: Tensor) -> float:
    #     """
    #     Perform a single step of the diffusion process.
    #
    #     :param x: The input value to be processed in this step.
    #     :return: The processed value after one step of diffusion.
    #     """
    #     beta_vec: Tensor = torch.ones(x.shape) * self.beta
    #     mean: Tensor = torch.sqrt(torch.ones(beta_vec.shape) - beta_vec) * x
    #     sdev: Tensor = torch.sqrt(beta_vec)
    #     return torch.normal(mean, sdev)


class MLP(torch.nn.Module):
    """Parameterize Denoising Process with MLP.

    As a map, this is (x_t, t) -> (mu_t, sigma_t)
    """

    def __init__(self, input_dim: int, time_dim, output_dim: int):
        super(MLP, self).__init__()

        self.encoder = torch.nn.Linear(time_dim, 1)
        self.tanh = torch.nn.Tanh()
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

        x = torch.nn.functional.normalize(x, dim=1)  # normalize input
        time_enc = self.encoder(time)
        x = torch.cat([x, time_enc], dim=1)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # Ensure output is non-negative for sigma
        # x[:, 1] = torch.exp(x[:, 1])
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

    loss = torch.sum(torch.square(mu1 - mu2)) / (2 * torch.square(sigma))
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


def main() -> None:

    tracker: Live = Live(report="md", monitor_system=True)

    time_dim: int = 1000  # one-hot-encoding of time steps
    tracker.log_param("Diffusion Time Steps", time_dim)

    sample_dim: int = 1
    output_dim: int = 1  # mu

    diffusor: Diffusor = Diffusor()
    diff_mlp = MLP(input_dim=sample_dim, time_dim=time_dim, output_dim=output_dim)
    optimizer: torch.optim.Optimizer = torch.optim.SGD(diff_mlp.parameters())

    data_0: torch.Tensor = gen_data()
    data0_img = track_hist_as_image(data_0, "Data Distribution.png", tracker)

    diff_mlp = train(
        diffusor,
        diff_mlp,
        optimizer,
        tracker=tracker,
        data_0=data_0,
        num_train_steps=100,
    )

    # generate samples
    num_samples: int = 1000
    gen_samples: torch.Tensor = generate_samples(
        diff_mlp, num_samples, diffusor.beta_schedule, save_samples=False
    )
    track_hist_as_image(gen_samples, "Generated Samples.png", tracker)

    tracker.end()


def track_hist_as_image(data: torch.Tensor, name: str, tracker: Live):
    plt.hist(data.detach().numpy(), bins=int(data.shape[0] / 100), density=True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    img = Image.open(buf)  # don't forget to close
    tracker.log_image(name, img)
    buf.close()
    img.close()


def gen_data() -> torch.Tensor:
    """Bimodal normal distribution."""
    n1: torch.distributions.Distribution = torch.distributions.Normal(-5, 1)
    n2: torch.distributions.Distribution = torch.distributions.Normal(5, 0.5)
    samples: torch.Tensor = torch.cat([n1.sample((10000,)), n2.sample((10000,))])
    return samples


def generate_samples(
    denoiser: MLP, num_samples: int, beta_schedule: Tensor, save_samples: bool = False
) -> torch.Tensor:
    """Generate samples from the denoiser model."""
    denoiser.eval()
    time_steps: int = beta_schedule.shape[
        0
    ]  # number of time steps in the diffusion process
    noise: Tensor = torch.randn((num_samples,))  # +1 for sample_dim

    if save_samples:
        samples: list[Tensor] = []
        samples.append(noise)

    for denoise_step in range(time_steps - 1, 0, -1):
        time_vec: Tensor = torch.zeros((num_samples, time_steps))
        time_vec[:, denoise_step] = 1.0
        noise_param = denoiser(noise.unsqueeze(1), time_vec)
        # print("Noise param shape and values:", end=" ")
        # print(noise_param.shape, noise_param)
        mu_out = noise_param[:, 0]
        sigma_out = beta_schedule[denoise_step]
        noise = mu_out + sigma_out * torch.randn_like(mu_out)
        if save_samples:
            samples.append(noise)

    return noise if not save_samples else torch.stack(samples)


def train(
    diffusor: Diffusor,
    denoiser: MLP,
    optimizer: torch.optim.Optimizer,
    data_0: torch.Tensor,
    tracker: Live,
    num_train_steps: int = 1000,
    batch_size: int = 64,
) -> MLP:
    """Training diffusor model."""

    tracker.log_param("Batch Size", batch_size)
    tracker.log_param("Num Training Steps", num_train_steps)
    tracker.log_param("Dataset Size", data_0.shape[0])

    # sample a time step t uniformly
    diff_steps: int = diffusor.diff_steps
    beta_schedule: torch.Tensor = diffusor.beta_schedule
    alpha_schedule: Tensor = diffusor.alpha_schedule

    loss_avg: float = 0.0

    for train_step in range(num_train_steps):

        # get batch and time step
        batch_idcs: torch.Tensor = torch.randint(0, data_0.shape[0], (batch_size,))
        batch: torch.Tensor = data_0[batch_idcs]
        time_step: int = torch.randint(1, diff_steps - 1, size=(1,)).item()
        data_t: Tensor = diffusor.n_step(batch, time_step)
        time_vec = torch.zeros((batch.shape[0], diff_steps))
        time_vec[:, 0] = 1.0

        optimizer.zero_grad()
        if time_step == 1:
            mu_out = denoiser(data_t.unsqueeze(1), time_vec)
            sigma_out = diffusor.beta_schedule[time_step].repeat(batch.shape[0], 1)
            likelihood = torch.distributions.Normal(mu_out, sigma_out).log_prob(batch)
            loss = -torch.mean(likelihood)
            # loss = -loss  # maximize negative log likelihood

        else:
            # t is in the middle of the diffusion process --> consistency loss

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

            # loss = kl_div_different_normal(q_mu, p_mu, q_sigma, p_sigma).mean()
            loss = kl_div_normal(q_mu, out, q_sigma).mean()

            # maximize loss
            # loss = -loss

        assert not loss.isnan(), "Loss is NaN"

        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
        optimizer.step()

        loss_avg = (train_step / (train_step + 1)) * loss_avg + (
            1 / (train_step + 1)
        ) * loss.item()

        if abs(loss.item() / loss_avg) < 100:
            tracker.log_metric("Loss", loss.item())

        total_param_norm = 0.0
        for param in denoiser.parameters():
            total_param_norm += param.norm().item()

        tracker.log_metric("Total parameter norm", total_param_norm)

        tracker.next_step()

    return denoiser


if __name__ == "__main__":
    main()
