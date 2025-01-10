"""
Code for reproducing the results of TABLE 1.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model.esn_classifier import ESNClassifier
from torchattacks import PGD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm

batch_size = 128
lr = 0.01
weight_decay = 3e-4
num_epochs = 10

radius_ls = [0.1, 0.5, 0.9, 1, 1.5]
with open("output/esn_mnist.txt", "w") as f:
    for radius in radius_ls:
        f.write(f"radius: {radius}\n")
        print(f"radius: {radius}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        esn_model = ESNClassifier(28, 500, 10, radius, 0.1, 0.1, 28, device).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            params=[
                {"params": esn_model.classifier.parameters()},
                {"params": esn_model.readout.parameters()}
            ]
            , lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * num_epochs), int(0.75 * num_epochs)])

        # data
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=False, transform=train_transform)
        test_set = datasets.MNIST(root='./data', train=False, download=False, transform=test_transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        # train
        for i in range(num_epochs):
            esn_model.train()
            for inputs, targets in tqdm.tqdm(train_loader):
                inputs = inputs.to(device)
                inputs = inputs.squeeze(1)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = esn_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()
            print(f"epoch: {i}, loss: {loss.item()}")
            f.write(f"epoch: {i}, loss: {loss.item()}\n")

            # validate
            esn_model.eval()
            total = 0
            correct = 0
            for j, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                inputs = inputs.squeeze(1)
                targets = targets.to(device)
                outputs = esn_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            print(f"epoch: {i}, accuracy: {correct / total}")
            f.write(f"epoch: {i}, accuracy: {correct / total}\n")

        # adversarial attack
        eps = 4/255
        pgd = PGD(model=esn_model, alpha=2 / 255, steps=20, eps=eps, random_start=True)
        total = 0
        correct = 0
        for j, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            inputs = inputs.squeeze(1)
            targets = targets.to(device)
            adv_inputs = pgd(inputs, targets)
            outputs = esn_model(adv_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        print(f"Accuracy under PGD attack with eps={eps}: {correct / total}")
        f.write(f"Accuracy under PGD attack with eps={eps}: {correct / total}\n")

        # save the model
        torch.save(esn_model.state_dict(), f"output/esn_mnist_{radius}.pth")


