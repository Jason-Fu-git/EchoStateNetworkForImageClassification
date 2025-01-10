"""
Code for producing Fig. 5 in the paper. Note : consider the randomness of 
the initialization, one may need to run the code multiple times to get the
same result as in the paper.
"""

from model.esn import ESN
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("bmh")

inputs1 = torch.randn(128, 500, 2) + torch.tensor([[4, 2]])
inputs2 = torch.randn(128, 500, 2) + torch.tensor([[2, 4]])
# plot the inputs
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs[0, 0].scatter(inputs1[:, 0, 0], inputs1[:, 0, 1])
axs[0, 0].scatter(inputs2[:, 0, 0], inputs2[:, 0, 1])
axs[0, 0].set_title("input")

esn_layer = ESN(2, 2, 2,0.9, 1, 1, 500)
h1 = esn_layer.forward_with_record(inputs1).detach().numpy()
h2 = esn_layer.forward_with_record(inputs2).detach().numpy()
# plot the states at timestep 10, 50 ,100 ,200 , 499

axs[0, 1].scatter(h1[:, 10, 0], h1[:, 10, 1])
axs[0, 1].scatter(h2[:, 10, 0], h2[:, 10, 1])
axs[0, 1].set_title("timestep 10")

axs[0, 2].scatter(h1[:, 50, 0], h1[:, 50, 1])
axs[0, 2].scatter(h2[:, 50, 0], h2[:, 50, 1])
axs[0, 2].set_title("timestep 50")

axs[1, 0].scatter(h1[:, 100, 0], h1[:, 100, 1])
axs[1, 0].scatter(h2[:, 100, 0], h2[:, 100, 1])
axs[1, 0].set_title("timestep 100")

axs[1, 1].scatter(h1[:, 200, 0], h1[:, 200, 1])
axs[1, 1].scatter(h2[:, 200, 0], h2[:, 200, 1])
axs[1, 1].set_title("timestep 200")

axs[1, 2].scatter(h1[:, 499, 0], h1[:, 499, 1])
axs[1, 2].scatter(h2[:, 499, 0], h2[:, 499, 1])
axs[1, 2].set_title("timestep 499")

plt.savefig("output/point_distance.png")
plt.show()

