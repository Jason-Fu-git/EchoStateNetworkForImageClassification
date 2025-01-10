"""
Test the echo state property of the ESN model, i.e. the hidden state should be the same if the input is the same.
"""

from model.esn import ESN
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("bmh")

x = torch.randn(1, 500, 100)
print(x.shape)
esn_layer = ESN(100, 500, 0.9, 0.05, 0.05, 500)
h1 = esn_layer.forward_with_record(x)
h2 = esn_layer.forward_with_record(x)
plt.scatter(np.arange(500), h1[0][0].detach().numpy())
plt.scatter(np.arange(500), h2[0][0].detach().numpy())
plt.show()
plt.scatter(np.arange(500), h1[-1][0].detach().numpy())
plt.scatter(np.arange(500), h2[-1][0].detach().numpy())
plt.show()