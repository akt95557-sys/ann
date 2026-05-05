#TensorFlow/Pytorch implementation of CNN
import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms as transforms

# Dataset
train = torchvision.datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test = torchvision.datasets.MNIST(
    './data',
    train=False,
    transform=transforms.ToTensor()
)

trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)

# CNN Model
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, 1, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, 1, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
opt = Adam(model.parameters(), lr=0.001)

# Training
for e in range(3):
    correct = 0
    total = 0

    for x, y in trainloader:
        out = model(x)
        loss = loss_fn(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        pred = out.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    print("Epoch", e+1, "Train Accuracy:", 100 * correct / total)

# Testing
correct = 0
total = 0

with torch.no_grad():
    for x, y in testloader:
        pred = model(x).argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()

print("Test Accuracy:", 100 * correct / total)