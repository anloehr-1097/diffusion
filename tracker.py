import torch
import matplotlib.pyplot as plt
import io
from PIL import Image
from dvclive import Live


def track_hist_as_image(data: torch.Tensor, name: str, tracker: Live):
    plt.hist(data.detach().numpy(), bins=int(data.shape[0] / 100), density=True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    img = Image.open(buf)  # don't forget to close
    tracker.log_image(name, img)
    buf.close()
    img.close()
