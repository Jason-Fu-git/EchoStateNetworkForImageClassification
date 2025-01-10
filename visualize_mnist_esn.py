"""
Code for producing Fig. 4 in the paper.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from model.esn_classifier import ESNClassifier
from torchattacks import PGD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("bmh")

esn_model = ESNClassifier(28, 500, 10, 0.9, 0.1, 0.1, 28, "cpu")
state_dict = torch.load("output/esn_mnist.pth", map_location="cpu")
esn_model.load_state_dict(state_dict)

# data
test_transform = transforms.Compose([
    transforms.ToTensor()
])
test_set = datasets.MNIST(root='./data', train=False, download=False, transform=test_transform)

# find two images with the same label
indices_0 = []
indices_1 = []
for i in range(len(test_set)):
    if test_set[i][1] == 0:
        indices_0.append(i)
    elif test_set[i][1] == 1:
        indices_1.append(i)
    if len(indices_0) >= 2 and len(indices_1) >= 2:
        break

# visualize
fig, axs = plt.subplots(4, 1, figsize=(5, 10))
for i, idx in enumerate(indices_0[:2] + indices_1[:2]):
    # original image
    img, label = test_set[idx]
    axs[i].imshow(img.squeeze(0), cmap="gray")
    axs[i].axis("off")
plt.show()

# forward pass
esn_model.eval()
img1, _ = test_set[indices_0[0]]
img2, _ = test_set[indices_0[1]]
img3, _ = test_set[indices_1[0]]
img4, _ = test_set[indices_1[1]]
img1 = img1.squeeze(1)
img2 = img2.squeeze(1)
img3 = img3.squeeze(1)
img4 = img4.squeeze(1)
img1 = img1.to("cpu")
img2 = img2.to("cpu")
img3 = img3.to("cpu")
img4 = img4.to("cpu")
_, h1_ls = esn_model.forward_with_information(img1)
_, h2_ls = esn_model.forward_with_information(img2)
_, h3_ls = esn_model.forward_with_information(img3)
_, h4_ls = esn_model.forward_with_information(img4)


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# visualize the hidden states
axs[0].scatter(range(500), h1_ls[-1].detach().numpy(), s=1, label="0")
axs[0].scatter(range(500), h3_ls[-1].detach().numpy(), s=1, label="1")
axs[0].set_xlabel("hidden unit")
axs[0].set_ylabel("value")
axs[0].set_title("Hidden States")
axs[0].legend()

# convert list to np.array
h1_ls = torch.stack(h1_ls).squeeze(1).detach().numpy()
h2_ls = torch.stack(h2_ls).squeeze(1).detach().numpy()
h3_ls = torch.stack(h3_ls).squeeze(1).detach().numpy()
h4_ls = torch.stack(h4_ls).squeeze(1).detach().numpy()


def cos_sim(a, b, axis):
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


# plot the distance between the hidden states
axs[1].plot(np.arange(28), cos_sim(h1_ls, h2_ls, axis=1), label="0")
axs[1].plot(np.arange(28), cos_sim(h3_ls, h4_ls, axis=1), label="1")
axs[1].plot(np.arange(28), cos_sim(h1_ls, h3_ls, axis=1), label="0-1")
axs[1].set_xlabel("time step")
axs[1].set_ylabel("cosine similarity")
axs[1].set_title("Cosine Similarity between Hidden States")
axs[1].legend()

# save the plot
plt.savefig("output/cosine_similarity.png")
plt.show()

