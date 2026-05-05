#How to Train a Neural Network with TensorFlow/Pytorch and evaluation of logistic regression using tensorflow.
import torch
import torch.nn as nn
import torch.optim as optim

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

X = torch.tensor([[0.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 0.0],
                   [1.0, 1.0]])

y = torch.tensor([[0.0],
                   [1.0],
                   [1.0],
                   [0.0]])

model = LogisticRegressionModel()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 10000

for epoch in range(epochs):

    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    predicted = model(X)
    predicted = (predicted > 0.5).float()

    accuracy = (predicted.eq(y).sum().item()) / y.size(0)

print(f'Accuracy: {accuracy * 100:.2f}%')

print("\nPredictions after training:")
print(predicted)