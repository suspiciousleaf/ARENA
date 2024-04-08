# %%

import torch as t
import einops
import tqdm

# import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


dataset = MNIST(root="data", download=True, transform=ToTensor())


def make_img_1d(imgs: t.Tensor) -> t.Tensor:
    return einops.rearrange(imgs, "... h w -> ... (h w)")


def one_hot_encoding(i: int, num_classes: int) -> t.Tensor:
    result = t.zeros([num_classes])
    result[i] = 1
    return result


training_imgs = []
expected_outputs_in_training = []
non_training_imgs = []
expected_outputs_in_non_training = []
counter = 0
total_imgs = 2000
num_of_training_imgs = 1000
for img, label in dataset:
    if counter >= total_imgs:
        break
    if counter < num_of_training_imgs:
        training_imgs.append(make_img_1d(img).squeeze())
        expected_outputs_in_training.append(one_hot_encoding(label, num_classes=10))
    else:
        non_training_imgs.append(make_img_1d(img).squeeze())
        expected_outputs_in_non_training.append(one_hot_encoding(label, num_classes=10))
    counter += 1

training_imgs = t.stack(training_imgs)
expected_outputs_in_training = t.stack(expected_outputs_in_training)

# %%


class SimpleNeuralNet(t.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.implementation = t.nn.Sequential(
            t.nn.Linear(in_features=784, out_features=128),
            t.nn.ReLU(),
            t.nn.Linear(in_features=128, out_features=40),
            t.nn.ReLU(),
            t.nn.Linear(in_features=40, out_features=10),
            t.nn.Softmax(dim=-1),
        )

    def forward(self, t: t.Tensor):
        return self.implementation(t)


def train(model: SimpleNeuralNet, epochs: int, lr: int):
    optimizer = t.optim.SGD(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        output = model(training_imgs)
        # For those who are confused why we use MSE loss here for a
        # classification task, see https://arxiv.org/abs/2006.07322
        loss = t.nn.functional.mse_loss(output, expected_outputs_in_training)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0:
            print(f"Initial loss: {loss=}")
        elif epoch == epochs - 1:
            print(f"Final loss: {loss=}")


model = SimpleNeuralNet()


train(model, epochs=100, lr=10)


# Let's look at an image that wasn't part of the training data

total_guesses = 1000
correct_guesses = 0
guesses_made = 0


for i in range(total_guesses):
    try:
        non_training_img_idx = i
        img_outside_of_training_dataset = non_training_imgs[non_training_img_idx]
        label = expected_outputs_in_non_training[non_training_img_idx].argmax()

        # print(f"Expected label: {label}")
        # plt.imshow(einops.rearrange(img_outside_of_training_dataset, "(h w) -> h w", h=28))

        model_all_guesses = model(img_outside_of_training_dataset)
        model_guess_highest_prob = model(img_outside_of_training_dataset).argmax()
        if label == model_guess_highest_prob:
            correct_guesses += 1
        guesses_made += 1
    except:
        break

    # print(f"Model guessed this was: {model_guess_highest_prob}")

print(
    f"Accuracy: {correct_guesses} / {guesses_made}, {correct_guesses/guesses_made*100}%"
)

# [784,4000,400,10] Ep 100 LR 10: Accuracy: 88.6%
# [784,4000,400,10] Ep 500 LR 10: Accuracy: 9.8%
# [784,4000,400,10] Ep 500 LR 2: Accuracy: 87.7%
# [784,4000,400,10] Ep 1000 LR 1: Accuracy: 88.0%

# [784,2000,400,10] Ep 100 LR 10: Accuracy: 82.3%

# [784,200,40,10] Ep 100 LR 10: Accuracy: 86.8%
# [784,200,40,10] Ep 1000 LR 1: Accuracy: 86.0%
# [784,200,40,10] Ep 10000 LR 1: Accuracy: 86.5%
# [784,200,40,10] Ep 100 LR 5: Accuracy: 77.9%

# [784,100,40,10] Ep 100 LR 5: Accuracy: 81.2%
# [784,100,40,10] Ep 1000 LR 5: Accuracy: 86.8%
# [784,100,40,10] Ep 1000 LR 2: Accuracy: 86.4%
# [784,100,40,10] Ep 10000 LR 5: Accuracy: 87.0%

# %%
