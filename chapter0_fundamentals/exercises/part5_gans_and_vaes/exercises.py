# %%
import os
import sys
import torch as t
from torch import nn, optim
import einops
from einops.layers.torch import Rearrange
from tqdm import tqdm
from dataclasses import dataclass, field
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from typing import Optional, Tuple, List, Literal, Union
import plotly.express as px
import torchinfo
import time
import wandb
from PIL import Image
import pandas as pd
from pathlib import Path
from datasets import load_dataset

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part5_gans_and_vaes"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part2_cnns.utils import print_param_count
import part5_gans_and_vaes.tests as tests
import part5_gans_and_vaes.solutions as solutions
from plotly_utils import imshow

from part2_cnns.solutions import (
    Linear,
    ReLU,
    Sequential,
    BatchNorm2d,
)
from part2_cnns.solutions_bonus import (
    pad1d,
    pad2d,
    conv1d_minimal,
    conv2d_minimal,
    Conv2d,
    Pair,
    IntOrPair,
    force_pair,
)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


# %%
class ConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
    ):
        """
        Same as torch.nn.ConvTranspose2d with bias=False.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_size = force_pair(kernel_size)
        sf = 1 / (self.out_channels * kernel_size[0] * kernel_size[1]) ** 0.5

        self.weight = nn.Parameter(
            sf * (2 * t.rand(in_channels, out_channels, *kernel_size) - 1)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return solutions.conv_transpose2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"{key}={getattr(self, key)}"
                for key in [
                    "in_channels",
                    "out_channels",
                    "kernel_size",
                    "stride",
                    "padding",
                ]
            ]
        )


# %%
class Tanh(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.tanh(x)


tests.test_Tanh(Tanh)


# %%
class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x.maximum(x.new_zeros(1)) + self.negative_slope * x.minimum(
            x.new_zeros(1)
        )

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


tests.test_LeakyReLU(LeakyReLU)


# %%
class Sigmoid(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x.sigmoid()


tests.test_Sigmoid(Sigmoid)

# %%


class Generator(nn.Module):

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):
        """
        Implements the generator architecture from the DCGAN paper (the diagram at the top
        of page 4). We assume the size of the activations doubles at each layer (so image
        size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            latent_dim_size:
                the size of the latent dimension, i.e. the input to the generator
            img_size:
                the size of the image, i.e. the output of the generator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the generator (starting from
                the smallest / closest to the generated images, and working backwards to the
                latent vector).

        """
        super().__init__()

        n_layers = len(hidden_channels)
        assert (
            img_size % (2**n_layers) == 0
        ), "activation size must double at each layer"
        initial_img_size = img_size // (2**n_layers)

        all_in_channels = list(reversed(hidden_channels))[:-1]
        all_out_channels = list(reversed(hidden_channels))[1:]

        blocks = (
            [
                # start with (batch_size, latent_dim_size)
                Linear(
                    latent_dim_size,
                    (hidden_channels[-1] * initial_img_size**2),
                    bias=False,
                ),
                ## now (batch_size, hidden_channels[-1] * initial_img_size ** 2)
                Rearrange(
                    "b (c h w) -> b c h w", c=hidden_channels[-1], h=initial_img_size
                ),
                ## now (batch_size, hidden_channels[-1], initial_img_size, initial_img_size)
                BatchNorm2d(hidden_channels[-1]),
                ReLU(),
            ]
            + [
                Sequential(
                    ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                    BatchNorm2d(out_channels),
                    ReLU(),
                )
                for in_channels, out_channels in zip(all_in_channels, all_out_channels)
            ]
            + [
                ConvTranspose2d(hidden_channels[0], img_channels, 4, 2, 1),
                Tanh(),
            ]
        )

        self.blocks = Sequential(*blocks)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.blocks(x)


class Discriminator(nn.Module):

    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):
        """
        Implements the discriminator architecture from the DCGAN paper (the mirror image of
        the diagram at the top of page 4). We assume the size of the activations doubles at
        each layer (so image size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            img_size:
                the size of the image, i.e. the input of the discriminator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the discriminator (starting from
                the smallest / closest to the input image, and working forwards to the probability
                output).
        """
        super().__init__()
        n_layers = len(hidden_channels)
        assert (
            img_size % (2**n_layers) == 0
        ), "activation size must double at each layer"

        all_in_channels = hidden_channels[:-1]
        all_out_channels = hidden_channels[1:]

        blocks = (
            [
                Conv2d(img_channels, hidden_channels[0], 4, 2, 1),
                LeakyReLU(),
            ]
            + [
                Sequential(
                    Conv2d(in_channels, out_channels, 4, 2, 1),
                    BatchNorm2d(out_channels),
                    LeakyReLU(),
                )
                for in_channels, out_channels in zip(all_in_channels, all_out_channels)
            ]
            + [
                Rearrange("b c h w -> b (c h w)"),
                Linear(
                    hidden_channels[-1]
                    * (img_size // (2 ** len(hidden_channels))) ** 2,
                    1,
                    bias=False,
                ),
                Sigmoid(),
            ]
        )
        self.blocks = Sequential(*blocks)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.blocks(x)


class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):
        """
        Implements the DCGAN architecture from the DCGAN paper (i.e. a combined generator
        and discriminator).
        """
        super().__init__()
        self.netD = Discriminator(img_size, img_channels, hidden_channels)
        self.netG = Generator(latent_dim_size, img_size, img_channels, hidden_channels)
        initialize_weights(self.netD)
        initialize_weights(self.netG)


def initialize_weights(model: nn.Module) -> None:
    """
    Initializes weights according to the DCGAN paper, by modifying model weights in place.
    """
    for _, m in model.named_modules():
        if (
            isinstance(m, Conv2d)
            or isinstance(m, ConvTranspose2d)
            or isinstance(m, Linear)
        ):
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, BatchNorm2d):
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.constant_(m.bias, 0)


tests.test_initialize_weights(
    initialize_weights, ConvTranspose2d, Conv2d, Linear, BatchNorm2d
)
# print_param_count(Generator(), solutions.DCGAN().netG)
# print_param_count(Discriminator(), solutions.DCGAN().netD)

# %%
model = DCGAN().to(device)
x = t.randn(3, 100).to(device)
statsG = torchinfo.summary(model.netG, input_data=x)
statsD = torchinfo.summary(model.netD, input_data=model.netG(x))
print(statsG, statsD)

# %%
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("nielsr/CelebA-faces")
print("Dataset loaded.")

# Create path to save the data
celeb_data_dir = section_dir / "data" / "celeba" / "img_align_celeba"
if not celeb_data_dir.exists():
    os.makedirs(celeb_data_dir)

    # Iterate over the dataset and save each image
    for idx, item in tqdm(
        enumerate(dataset["train"]),
        total=len(dataset["train"]),
        desc="Saving individual images...",
    ):
        # The image is already a JpegImageFile, so we can directly save it
        item["image"].save(
            exercises_dir
            / "part5_gans_and_vaes"
            / "data"
            / "celeba"
            / "img_align_celeba"
            / f"{idx:06}.jpg"
        )

    print("All images have been saved.")


# %%
def get_dataset(dataset: Literal["MNIST", "CELEB"], train: bool = True) -> Dataset:
    assert dataset in ["MNIST", "CELEB"]

    if dataset == "CELEB":
        image_size = 64
        assert train, "CelebA dataset only has a training set"
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        trainset = datasets.ImageFolder(
            root=exercises_dir / "part5_gans_and_vaes" / "data" / "celeba",
            transform=transform,
        )

    elif dataset == "MNIST":
        img_size = 28
        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        trainset = datasets.MNIST(
            root=exercises_dir / "part5_gans_and_vaes" / "data",
            transform=transform,
            download=True,
        )

    return trainset


def display_data(x: t.Tensor, nrows: int, title: str):
    """Displays a batch of data, using plotly."""
    # Reshape into the right shape for plotting (make it 2D if image is monochrome)
    y = einops.rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=nrows).squeeze()
    # Normalize, in the 0-1 range
    y = (y - y.min()) / (y.max() - y.min())
    # Display data
    imshow(
        y,
        binary_string=(y.ndim == 2),
        height=50 * (nrows + 5),
        title=title + f"<br>single input shape = {x[0].shape}",
    )


# Load in MNIST, get first batch from dataloader, and display
trainset_mnist = get_dataset("MNIST")
x = next(iter(DataLoader(trainset_mnist, batch_size=64)))[0]
display_data(x, nrows=8, title="MNIST data")

# Load in CelebA, get first batch from dataloader, and display
trainset_celeb = get_dataset("CELEB")
x = next(iter(DataLoader(trainset_celeb, batch_size=64)))[0]
display_data(x, nrows=8, title="CalebA data")


# %%
@dataclass
class DCGANArgs:
    """
    Class for the arguments to the DCGAN (training and architecture).
    Note, we use field(defaultfactory(...)) when our default value is a mutable object.
    """

    latent_dim_size: int = 100
    hidden_channels: List[int] = field(default_factory=lambda: [128, 256, 512])
    dataset: Literal["MNIST", "CELEB"] = "CELEB"
    batch_size: int = 64
    epochs: int = 3
    lr: float = 0.0002
    betas: Tuple[float] = (0.5, 0.999)
    seconds_between_eval: int = 20
    wandb_project: Optional[str] = "day5-gan"
    wandb_name: Optional[str] = None


class DCGANTrainer:
    def __init__(self, args: DCGANArgs):
        self.args = args

        self.trainset = get_dataset(self.args.dataset)
        self.trainloader = DataLoader(
            self.trainset, batch_size=args.batch_size, shuffle=True
        )

        batch, img_channels, img_height, img_width = next(iter(self.trainloader))[
            0
        ].shape
        assert img_height == img_width

        self.model = (
            DCGAN(
                args.latent_dim_size,
                img_height,
                img_channels,
                args.hidden_channels,
            )
            .to(device)
            .train()
        )

        self.optG = t.optim.Adam(
            self.model.netG.parameters(), lr=args.lr, betas=args.betas
        )
        self.optD = t.optim.Adam(
            self.model.netD.parameters(), lr=args.lr, betas=args.betas
        )

    def training_step_discriminator(
        self, img_real: t.Tensor, img_fake: t.Tensor
    ) -> t.Tensor:
        """
        Given a batch of real images and a batch of fake images, performs a
        gradient step on the discriminator to maximize log(D(img_real)) +
        log(1-D(img_fake)).
        """
        self.optD.zero_grad()
        d_real = self.model.netD(img_real)
        d_fake = self.model.netD(img_fake)
        loss = -(t.log(d_real) + t.log(1 - d_fake)).mean()
        loss.backward()
        self.optD.step()
        return loss

    def training_step_generator(self, img_fake: t.Tensor) -> t.Tensor:
        """
        Performs a gradient step on the generator to maximize log(D(G(z))).
        """
        self.optG.zero_grad()
        # D_G_z = self.model.netD(img_fake)
        # labels_real = t.ones_like(D_G_z)
        # loss = nn.BCELoss()(D_G_z, labels_real)
        loss = -t.log(self.model.netD(img_fake)).mean()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.model.netG.parameters(), 1.0)
        self.optG.step()
        return loss

    @t.inference_mode()
    def evaluate(self) -> None:
        """
        Performs evaluation by generating 8 instances of random noise and passing them through
        the generator, then logging the results to Weights & Biases.
        """
        noise = t.FloatTensor(8, self.args.latent_dim_size).to(device).uniform_(0, 1)
        images = self.model.netG(noise)
        # display_data(images, 8, 'Generated celebrities')

    def train(self) -> None:
        """
        Performs a full training run, while logging to Weights & Biases.
        """
        self.step = 0
        last_log_time = time.time()
        # wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
        # wandb.watch(self.model)

        train_seq = 0
        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=len(self.trainloader))

            for img_real, label in progress_bar:

                # Generate random noise & fake image
                noise = t.randn(self.args.batch_size, self.args.latent_dim_size).to(
                    device
                )
                img_real = img_real.to(device)
                img_fake = self.model.netG(noise)

                # Training steps
                if train_seq % 1 == 0:
                    lossD = self.training_step_discriminator(
                        img_real, img_fake.detach()
                    )
                else:
                    lossD = 0
                lossG = self.training_step_generator(img_fake)
                train_seq += 1

                # Log data
                # wandb.log(dict(lossD=lossD, lossG=lossG), step=self.step)

                # Update progress bar
                self.step += img_real.shape[0]
                progress_bar.set_description(
                    f"{epoch=}, lossD={lossD:.4f}, lossG={lossG:.4f}, examples_seen={self.step}"
                )

                # Evaluate model on the same batch of random data
                if time.time() - last_log_time > self.args.seconds_between_eval:
                    last_log_time = time.time()
                    self.evaluate()

        # wandb.finish()


# %%
if False:
    # Arguments for MNIST
    args = DCGANArgs(
        dataset="MNIST",
        hidden_channels=[32, 64],
        epochs=15,
        batch_size=32,
        seconds_between_eval=20,
    )
    trainer = DCGANTrainer(args)
    trainer.train()

# %%

if False:
    # Arguments for CelebA
    args = DCGANArgs(
        dataset="CELEB",
        hidden_channels=[128, 256, 512],
        batch_size=8,
        epochs=3,
        seconds_between_eval=30,
    )
    trainer = DCGANTrainer(args)
    trainer.train()


# %%


class Autoencoder(nn.Module):

    def __init__(self, latent_dim_size: int, hidden_dim_size: int):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size
        self.encoder = Sequential(
            Conv2d(1, 16, 4, stride=2, padding=1),
            ReLU(),
            Conv2d(16, 32, 4, stride=2, padding=1),
            ReLU(),
            Rearrange("b c h w -> b (c h w)"),
            Linear(7 * 7 * 32, hidden_dim_size),
            ReLU(),
            Linear(hidden_dim_size, latent_dim_size),
        )
        self.decoder = Sequential(
            Linear(latent_dim_size, hidden_dim_size),
            ReLU(),
            Linear(hidden_dim_size, 7 * 7 * 32),
            ReLU(),
            Rearrange("b (c h w) -> b c w h", c=32, h=7, w=7),
            ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        z = self.encoder(x)
        x_prime = self.decoder(z)
        return x_prime


# %%
soln_Autoencoder = solutions.Autoencoder(latent_dim_size=5, hidden_dim_size=128)
my_Autoencoder = Autoencoder(latent_dim_size=5, hidden_dim_size=128)

print_param_count(my_Autoencoder, soln_Autoencoder)
# %%
testset = get_dataset("MNIST", train=False)
HOLDOUT_DATA = dict()
for data, target in DataLoader(testset, batch_size=1):
    if target.item() not in HOLDOUT_DATA:
        HOLDOUT_DATA[target.item()] = data.squeeze()
        if len(HOLDOUT_DATA) == 10:
            break
HOLDOUT_DATA = (
    t.stack([HOLDOUT_DATA[i] for i in range(10)])
    .to(dtype=t.float, device=device)
    .unsqueeze(1)
)

display_data(HOLDOUT_DATA, nrows=2, title="MNIST holdout data")


# %%
@dataclass
class AutoencoderArgs:
    latent_dim_size: int = 5
    hidden_dim_size: int = 128
    dataset: Literal["MNIST", "CELEB"] = "MNIST"
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-3
    betas: Tuple[float] = (0.5, 0.999)
    seconds_between_eval: int = 5
    wandb_project: Optional[str] = "day5-ae-mnist"
    wandb_name: Optional[str] = None


class AutoencoderTrainer:
    def __init__(self, args: AutoencoderArgs):
        self.args = args
        self.trainset = get_dataset(args.dataset)
        self.trainloader = DataLoader(
            self.trainset, batch_size=args.batch_size, shuffle=True
        )
        self.model = Autoencoder(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(device)
        self.optimizer = t.optim.Adam(
            self.model.parameters(), lr=args.lr, betas=args.betas
        )

    def training_step(self, imgs: t.Tensor) -> t.Tensor:
        """
        Performs a training step on the batch of images in `img`. Returns the loss.
        """
        out_imgs = self.model(imgs)
        loss = nn.MSELoss()(imgs, out_imgs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    @t.inference_mode()
    def evaluate(self) -> None:
        """
        Evaluates model on holdout data, logs to weights & biases.
        """
        display_data(
            self.model(HOLDOUT_DATA), nrows=2, title="MNIST holdout data, encoded"
        )

    def train(self) -> None:
        """
        Performs a full training run, logging to wandb.
        """
        self.step = 0
        last_log_time = time.time()
        # wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
        # wandb.watch(self.model)

        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=int(len(self.trainloader)))

            for i, (img, label) in enumerate(
                progress_bar
            ):  # remember that label is not used

                img = img.to(device)
                loss = self.training_step(img)
                # wandb.log(dict(loss=loss), step=self.step)

                # Update progress bar
                self.step += img.shape[0]
                progress_bar.set_description(
                    f"{epoch=}, {loss=:.4f}, examples_seen={self.step}"
                )

                # Evaluate model on the same holdout data
                if time.time() - last_log_time > self.args.seconds_between_eval:
                    last_log_time = time.time()
                    self.evaluate()

        # wandb.finish()


args = AutoencoderArgs()
trainer = AutoencoderTrainer(args)
trainer.train()


# %%
@t.inference_mode()
def visualise_output(
    model: Autoencoder,
    n_points: int = 11,
    interpolation_range: Tuple[float, float] = (-3, 3),
) -> None:
    """
    Visualizes the output of the decoder, along the first two latent dims.
    """
    # Constructing latent dim data by making two of the dimensions vary indep in the interpolation range
    grid_latent = t.zeros(n_points**2, model.latent_dim_size).to(device)
    x = t.linspace(*interpolation_range, n_points).to(device)
    grid_latent[:, 0] = einops.repeat(x, "dim1 -> (dim1 dim2)", dim2=n_points)
    grid_latent[:, 1] = einops.repeat(x, "dim2 -> (dim1 dim2)", dim1=n_points)

    # Pass through decoder
    output = model.decoder(grid_latent).cpu().numpy()

    # Normalize & truncate, then unflatten back into a grid shape
    output_truncated = np.clip((output * 0.3081) + 0.1307, 0, 1)
    output_single_image = einops.rearrange(
        output_truncated,
        "(dim1 dim2) 1 height width -> (dim1 height) (dim2 width)",
        dim1=n_points,
    )

    # Display the results
    px.imshow(
        output_single_image,
        color_continuous_scale="greys_r",
        title="Decoder output from varying first principal components of latent space",
    ).update_layout(
        xaxis=dict(
            title_text="dim1",
            tickmode="array",
            tickvals=list(range(14, 14 + 28 * n_points, 28)),
            ticktext=[f"{i:.2f}" for i in x],
        ),
        yaxis=dict(
            title_text="dim2",
            tickmode="array",
            tickvals=list(range(14, 14 + 28 * n_points, 28)),
            ticktext=[f"{i:.2f}" for i in x],
        ),
    ).show()


visualise_output(trainer.model)


# %%
@t.inference_mode()
def visualise_input(model: Autoencoder, dataset: Dataset, vae: bool) -> None:
    """
    Visualises (in the form of a scatter plot) the input data in the latent space, along the first two dims.
    """
    # First get the model images' latent vectors, along first 2 dims
    imgs = t.stack([batch for batch, label in dataset]).to(device)
    encode = lambda imgs: model.encoder(imgs)[0] if vae else model.encoder
    latent_vectors = encode(imgs)
    # if latent_vectors.ndim == 3: latent_vectors = latent_vectors[0] # useful for VAEs later
    latent_vectors = latent_vectors[:, :2].cpu().numpy()
    labels = [str(label) for img, label in dataset]

    # Make a dataframe for scatter (px.scatter is more convenient to use when supplied with a dataframe)
    df = pd.DataFrame(
        {"dim1": latent_vectors[:, 0], "dim2": latent_vectors[:, 1], "label": labels}
    )
    df = df.sort_values(by="label")
    fig = px.scatter(df, x="dim1", y="dim2", color="label")
    fig.update_layout(
        height=700,
        width=700,
        title="Scatter plot of latent space dims",
        legend_title="Digit",
    )
    data_range = df["dim1"].max() - df["dim1"].min()

    # Add images to the scatter plot (optional)
    output_on_data_to_plot = encode(HOLDOUT_DATA.to(device))[:, :2].cpu()
    print(output_on_data_to_plot.shape)
    # if output_on_data_to_plot.ndim == 3: output_on_data_to_plot = output_on_data_to_plot[0] # useful for VAEs; see later
    print(output_on_data_to_plot.shape)
    data_translated = (HOLDOUT_DATA.cpu().numpy() * 0.3081) + 0.1307
    data_translated = (255 * data_translated).astype(np.uint8).squeeze()
    for i in range(10):
        x, y = output_on_data_to_plot[i]
        fig.add_layout_image(
            source=Image.fromarray(data_translated[i]).convert("L"),
            xref="x",
            yref="y",
            x=x,
            y=y,
            xanchor="right",
            yanchor="top",
            sizex=data_range / 15,
            sizey=data_range / 15,
        )
    fig.show()


small_dataset = Subset(get_dataset("MNIST"), indices=range(0, 5000))
visualise_input(trainer.model, small_dataset, True)


# %%
class VAE(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    def __init__(self, latent_dim_size: int, hidden_dim_size: int):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, hidden_dim_size),
            nn.ReLU(),
            nn.Linear(hidden_dim_size, latent_dim_size * 2),
            Rearrange(
                "b (n latent_dim) -> n b latent_dim", n=2
            ),  # makes it easier to separate mu and sigma
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, hidden_dim_size),
            nn.ReLU(),
            nn.Linear(hidden_dim_size, 7 * 7 * 32),
            nn.ReLU(),
            Rearrange("b (c h w) -> b c w h", c=32, h=7, w=7),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def sample_latent_vector(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        """
        Passes `x` through the encoder. Returns the mean and log std dev of the latent vector,
        as well as the latent vector itself. This function can be used in `forward`, but also
        used on its own to generate samples for evaluation.
        """
        mu, logsigma = self.encoder(x)
        sigma = t.exp(logsigma)
        z = mu + sigma * t.randn_like(mu)
        return z, mu, logsigma

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        """
        Passes `x` through the encoder and decoder. Returns the reconstructed input, as well
        as mu and logsigma.
        """
        z, mu, logsigma = self.sample_latent_vector(x)
        x_prime = self.decoder(z)
        return x_prime, mu, logsigma


# %%
@dataclass
class VAEArgs(AutoencoderArgs):
    wandb_project: Optional[str] = "day5-vae-mnist"
    beta_kl: float = 0.1


class VAETrainer:
    def __init__(self, args: VAEArgs):
        self.args = args
        self.trainset = get_dataset(args.dataset)
        self.trainloader = DataLoader(
            self.trainset, batch_size=args.batch_size, shuffle=True
        )
        self.model = VAE(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(device)
        self.optimizer = t.optim.Adam(
            self.model.parameters(), lr=args.lr, betas=args.betas
        )

    def training_step(self, imgs: t.Tensor) -> t.Tensor:
        """
        Performs a training step on the batch of images in `img`. Returns the loss.
        """
        out_imgs, mu, logsigma = self.model(imgs)
        sigma_squared = t.exp(2 * logsigma)
        loss = (
            nn.MSELoss()(imgs, out_imgs)
            + self.args.beta_kl * ((sigma_squared + mu**2 - 1) / 2 - logsigma).mean()
        )
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    @t.inference_mode()
    def evaluate(self) -> None:
        """
        Evaluates model on holdout data, logs to weights & biases.
        """
        display_data(
            self.model(HOLDOUT_DATA)[0], nrows=2, title="MNIST holdout data, encoded"
        )

    def train(self) -> None:
        """
        Performs a full training run, logging to wandb.
        """
        self.step = 0
        last_log_time = time.time()
        # wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
        # wandb.watch(self.model)

        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=int(len(self.trainloader)))

            for i, (img, label) in enumerate(
                progress_bar
            ):  # remember that label is not used

                img = img.to(device)
                loss = self.training_step(img)
                # wandb.log(dict(loss=loss), step=self.step)

                # Update progress bar
                self.step += img.shape[0]
                progress_bar.set_description(
                    f"{epoch=}, {loss=:.4f}, examples_seen={self.step}"
                )

                # Evaluate model on the same holdout data
                if time.time() - last_log_time > self.args.seconds_between_eval:
                    last_log_time = time.time()
                    self.evaluate()

        # wandb.finish()


args = VAEArgs(latent_dim_size=10, hidden_dim_size=100)
trainer = VAETrainer(args)
trainer.train()

# %%
visualise_output(trainer.model)

# %%
small_dataset = Subset(get_dataset("MNIST"), indices=range(0, 5000))
visualise_input(trainer.model, small_dataset)


# %%
