import torch
import numpy as np
from torch import Tensor
from dvclive import Live
from tracker import track_hist_as_image

from data import gen_data
from diffusor import Diffusor
from loss import kl_div_normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(torch.nn.Module):
    """Parameterize Denoising Process with MLP.

    As a map, this is (x_t, t) -> (mu_t, sigma_t)
    """

    def __init__(self, input_dim: int, time_dim, output_dim: int):
        super(MLP, self).__init__()

        self.encoder = torch.nn.Linear(time_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.fc1 = torch.nn.Linear(input_dim + 1, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, output_dim)
        self.relu = torch.nn.ReLU()
        torch.nn.init.normal_(self.fc1.weight, std=1e-2)
        torch.nn.init.normal_(self.fc2.weight, std=1e-2)
        torch.nn.init.normal_(self.fc3.weight, std=1e-2)

    def forward(self, x: Tensor, time: Tensor) -> Tensor:
        """
        Forward pass through the MLP.

        :param x: Input tensor.
        :param time: Time tensor for conditioning.
        :return: Output tensor after passing through the MLP.
        """
        # print(x.shape, time.shape)

        time_enc = self.encoder(time)
        x = torch.cat([x, time_enc], dim=1)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        # Ensure output is non-negative for sigma
        # x[:, 1] = torch.exp(x[:, 1])
        return x


def main() -> None:

    tracker: Live = Live(report="md", monitor_system=True)

    time_dim: int = 1000  # one-hot-encoding of time steps
    learning_rate: float = 1e-3
    tracker.log_param("Diffusion Time Steps", time_dim)
    tracker.log_param("Learning Rate", learning_rate)

    sample_dim: int = 1
    output_dim: int = 1  # mu

    diffusor: Diffusor = Diffusor()
    diff_mlp = MLP(input_dim=sample_dim, time_dim=time_dim, output_dim=output_dim)
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        diff_mlp.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    data_0: torch.Tensor = gen_data()
    data0_img = track_hist_as_image(data_0, "Data Distribution.png", tracker)

    diff_mlp = train(
        diffusor,
        diff_mlp,
        optimizer,
        tracker=tracker,
        data_0=data_0,
        num_train_steps=2000,
        batch_size=2048,
    )

    # generate samples
    num_samples: int = 1000
    gen_samples: torch.Tensor = generate_samples(
        diff_mlp, num_samples, diffusor.beta_schedule, save_samples=False
    )
    # print("Generated smaples shape: ", gen_samples.shape)
    track_hist_as_image(gen_samples, "Generated Samples.png", tracker)

    tracker.end()


def generate_samples(
    denoiser: MLP, num_samples: int, beta_schedule: Tensor, save_samples: bool = False
) -> torch.Tensor:
    """Generate samples from the denoiser model."""
    denoiser.to("cpu")
    denoiser.eval()
    time_steps: int = beta_schedule.shape[
        0
    ]  # number of time steps in the diffusion process
    alpha_schedule: Tensor = torch.ones(time_steps) - beta_schedule
    noise: Tensor = torch.randn((num_samples,))  # +1 for sample_dim

    if save_samples:
        samples: list[Tensor] = []
        samples.append(noise)

    for denoise_step in range(time_steps - 1, 0, -1):
        time_vec: Tensor = torch.zeros((num_samples, time_steps))
        time_vec[:, denoise_step] = 1.0
        noise_param = denoiser(noise.unsqueeze(1), time_vec)
        mu_out = noise_param[:, 0]

        # sigma_out = \tilde{\beta}_t
        alpha_bar_t_minus_1: Tensor = torch.prod(alpha_schedule[: denoise_step - 1])
        beta_t: Tensor = beta_schedule[denoise_step]
        alpha_bar_t: Tensor = torch.prod(alpha_schedule[:denoise_step])
        sigma_out = beta_t * (1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t)
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
    batch_size: int = 1024,
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

    denoiser.to(device)
    for train_step in range(num_train_steps):

        # get batch and time step
        batch_idcs: torch.Tensor = torch.randint(0, data_0.shape[0], (batch_size,))
        batch: torch.Tensor = data_0[batch_idcs]
        time_step: int = torch.randint(1, diff_steps - 1, size=(1,)).item()
        data_t: Tensor = diffusor.n_step(batch, time_step)
        time_vec = torch.zeros((batch.shape[0], diff_steps)).to(device)
        time_vec[:, time_step] = 1.0

        optimizer.zero_grad()
        if time_step == 1:
            # print("L_0")
            mu_out = denoiser(data_t.unsqueeze(1).to(device), time_vec)
            sigma_out = diffusor.beta_schedule[time_step].repeat(batch.shape[0], 1)
            # TODO can we differentiate this?
            likelihood = torch.distributions.Normal(
                mu_out, sigma_out.to(device)
            ).log_prob(batch.to(device))
            # print(f"Mu: {mu_out}, Sigma: {sigma_out}, likelihood: {likelihood}")
            loss = -torch.mean(likelihood)

        else:
            # print(f"L_{time_step}")
            # t is in the middle of the diffusion process --> consistency loss

            # parameters of q(x_{t-1} | x_t, x_0)
            alpha_bar_t_minus_1: Tensor = torch.prod(alpha_schedule[: time_step - 1])
            beta_t: Tensor = beta_schedule[time_step]
            alpha_bar_t: Tensor = torch.prod(alpha_schedule[:time_step])

            x_0_contrib: Tensor = (
                (torch.sqrt(alpha_bar_t_minus_1) * beta_t) / (1 - alpha_bar_t) * batch
            )
            x_t_contrib: Tensor = (
                torch.sqrt(alpha_bar_t)
                * (1 - alpha_bar_t_minus_1)
                / (1 - alpha_bar_t)
                * data_t
            )

            q_mu = x_0_contrib + x_t_contrib

            q_sigma = beta_t * (1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t)

            # parameters for p(x_{t-1} | x_t)
            out = denoiser(data_t.unsqueeze(1).to(device), time_vec)

            # loss = kl_div_different_normal(q_mu, p_mu, q_sigma, p_sigma).mean()
            loss = kl_div_normal(
                q_mu.detach().to(device), out.to(device), q_sigma.detach().to(device)
            ).mean()

            # print(
            #     f"loss_approx: {torch.sum(torch.square(out - q_mu.to(device)))}\n\
            #     p_mu: {torch.mean(out)},\n\
            #     q_mu: {torch.mean(q_mu)},\n\
            #     Sigma: {torch.mean(q_sigma)},\n\
            #     Loss: {loss.item()}\n"
            # )
            # maximize loss
            # loss = -loss

        assert not loss.isnan(), "Loss is NaN"

        # clip gradients to avoid exploding gradients
        loss.backward()
        grad_norm: Tensor = torch.nn.utils.clip_grad_norm_(
            denoiser.parameters(), max_norm=1.0
        )

        tracker.log_metric("Gradient norm", grad_norm.item())

        optimizer.step()

        loss_avg = (train_step / (train_step + 1)) * loss_avg + (
            1 / (train_step + 1)
        ) * loss.item()

        if abs(loss.item() / loss_avg) < 10:

            tracker.log_metric("Loss", loss.item())
        else:
            tracker.log_metric("Loss", loss_avg)

        total_param_norm = 0.0
        for param in denoiser.parameters():
            total_param_norm += param.norm().item()

        tracker.log_metric("Total parameter norm", total_param_norm)

        tracker.next_step()

    return denoiser


def print_model_summary(model: torch.nn.Module) -> None:
    """Print a summary of the model's architecture."""
    print("Model Summary:")
    print(model)
    print("\nModel Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
        print(torch.mean(param))
    print("\nTotal number of parameters:", sum(p.numel() for p in model.parameters()))
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )


def weights_init_uniform_rule(m):
    if isinstance(m, torch.nn.Linear):
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        torch.nn.init.uniform_(m.weight, -y, y)
        torch.nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    main()
