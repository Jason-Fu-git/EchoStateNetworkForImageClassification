"""
Code for producing Fig. 1 in the paper. Note : consider the randomness of 
the initialization, one may need to run the code multiple times to get the
same result as in the paper.
"""


from model.esn import ESN
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("bmh")

sequence_len = 100
batch_size = 128

sin_1_2 = np.sin(np.linspace(0, 2 * np.pi, sequence_len) * 1) + np.sin(np.linspace(0, 2 * np.pi, sequence_len) * 2)
sin_2_4 = np.sin(np.linspace(0, 2 * np.pi, sequence_len) * 2) + np.sin(np.linspace(0, 2 * np.pi, sequence_len) * 4)

# add noise
noise1 = torch.randn(batch_size, sequence_len, 1) * 0.1
noise2 = torch.randn(batch_size, sequence_len, 1) * 0.1
inputs1 = torch.tensor(sin_1_2).float().unsqueeze(-1).repeat(batch_size, 1, 1) + noise1
inputs2 = torch.tensor(sin_2_4).float().unsqueeze(-1).repeat(batch_size, 1, 1) + noise2

fig, axs = plt.subplots(2, 3, figsize=(15, 9))
# plot
axs[0, 0].plot(inputs1[0].squeeze().numpy())
axs[0, 0].plot(inputs2[0].squeeze().numpy())
axs[0, 0].set_title("input")
axs[0, 0].legend(["sin(1x) + sin(2x)", "sin(2x) + sin(4x)"])
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("f(t)")

esn_layer = ESN(1, 2,1, 0.9, 1,1, sequence_len)

h1 = esn_layer.forward_with_record(inputs1).detach().numpy()
h2 = esn_layer.forward_with_record(inputs2).detach().numpy()

# show the hidden states at timestep 0, 10, 20, 50, 99
axs[0,1].scatter(h1[:, 0, 0], h1[:, 0, 1], label="sin(1x) + sin(2x)")
axs[0,1].scatter(h2[:, 0, 0], h2[:, 0, 1], label="sin(2x) + sin(4x)")
axs[0,1].set_title("timestep 0")
axs[0,1].set_xlabel("x")
axs[0,1].set_ylabel("y")
axs[0,1].legend()

axs[0,2].scatter(h1[:, 10, 0], h1[:, 10, 1], label="sin(1x) + sin(2x)")
axs[0,2].scatter(h2[:, 10, 0], h2[:, 10, 1], label="sin(2x) + sin(4x)")
axs[0,2].set_title("timestep 10")
axs[0,2].set_xlabel("x")
axs[0,2].set_ylabel("y")
axs[0,2].legend()

axs[1,0].scatter(h1[:, 20, 0], h1[:, 20, 1], label="sin(1x) + sin(2x)")
axs[1,0].scatter(h2[:, 20, 0], h2[:, 20, 1], label="sin(2x) + sin(4x)")
axs[1,0].set_title("timestep 20")
axs[1,0].set_xlabel("x")
axs[1,0].set_ylabel("y")
axs[1,0].legend()

axs[1,1].scatter(h1[:, 50, 0], h1[:, 50, 1], label="sin(1x) + sin(2x)")
axs[1,1].scatter(h2[:, 50, 0], h2[:, 50, 1], label="sin(2x) + sin(4x)")
axs[1,1].set_title("timestep 50")
axs[1,1].set_xlabel("x")
axs[1,1].set_ylabel("y")
axs[1,1].legend()

axs[1,2].scatter(h1[:, 99, 0], h1[:, 99, 1], label="sin(1x) + sin(2x)")
axs[1,2].scatter(h2[:, 99, 0], h2[:, 99, 1], label="sin(2x) + sin(4x)")
axs[1,2].set_title("timestep 99")
axs[1,2].set_xlabel("x")
axs[1,2].set_ylabel("y")
axs[1,2].legend()

plt.savefig("output/sequence_distance.png")
plt.show()


