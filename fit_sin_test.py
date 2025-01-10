"""
Train an ESN to fit a sin function.
"""


from model.esn import ESN
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("bmh")

sequence_len = 1000

x = np.sin(4 * np.linspace(0, 2 * np.pi, sequence_len))
y = x ** 7
plt.plot(x)
plt.plot(y)
plt.show()

# reshape the data to (1, sequence_len, 1)
x = torch.tensor(x).float().unsqueeze(0).unsqueeze(-1)
y = torch.tensor(y).float().unsqueeze(0).unsqueeze(-1)

esn = ESN(1, 1000,  1, 0.9, 0.5, 0.5, sequence_len)
y_hat = esn.forward(x)
plt.plot(y_hat[0].detach().numpy().squeeze())
plt.plot(y[0].detach().numpy().squeeze())
plt.show()

# train the model
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(esn.linear.parameters(), lr=0.001, weight_decay=3e-4)
for i in range(100):
    optimizer.zero_grad()
    y_hat = esn.forward(x)
    loss = criterion(y_hat[:, 200:], y[:, 200:])
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f"loss: {loss.item()}")

y_hat = esn.forward(x)
plt.plot(y_hat[0].detach().numpy().squeeze())
plt.plot(y[0].detach().numpy().squeeze())
plt.show()



