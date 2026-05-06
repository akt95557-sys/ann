import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Iris Dataset
data = load_iris()
X = data.data
y = data.target

# One-hot encoding
Y = np.eye(3)[y]

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Softmax Function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Random Seed
np.random.seed(0)

# Network Architecture
input_neurons = 4
hidden_neurons = 10
output_neurons = 3

# Weight Initialization
W1 = np.random.randn(input_neurons, hidden_neurons) * 0.01
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons) * 0.01
b2 = np.zeros((1, output_neurons))

# Hyperparameters
learning_rate = 0.001
epochs = 5000

# Training
for epoch in range(epochs):

    # Forward Propagation
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    # Cross Entropy Loss
    loss = -np.mean(Y_train * np.log(A2 + 1e-8))

    # Backpropagation
    error = A2 - Y_train

    dW2 = np.dot(A1.T, error) / X_train.shape[0]
    db2 = np.sum(error, axis=0, keepdims=True) / X_train.shape[0]

    dA1 = np.dot(error, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = np.dot(X_train.T, dZ1) / X_train.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X_train.shape[0]

    # Update Weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print Loss
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing

Z1 = np.dot(X_test, W1) + b1
A1 = relu(Z1)

Z2 = np.dot(A1, W2) + b2
A2 = softmax(Z2)

# Predictions
predictions = np.argmax(A2, axis=1)
actual = np.argmax(Y_test, axis=1)

# Accuracy
accuracy = np.mean(predictions == actual)

print("\nPredicted Classes:")
print(predictions)

print("\nActual Classes:")
print(actual)

print(f"\nAccuracy: {accuracy * 100:.2f}%")