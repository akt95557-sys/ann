#Create a Neural network architecture from scratch in Python and use it to do multi-class classification on any data
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data
y = data.target

Y = np.eye(3)[y]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

np.random.seed(0)

W1 = np.random.randn(4, 5)
b1 = np.zeros((1, 5))
W2 = np.random.randn(5, 3)
b2 = np.zeros((1, 3))

lr = 0.01
epochs = 500

for _ in range(epochs):

    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    error = A2 - Y_train

    dW2 = np.dot(A1.T, error)
    db2 = np.sum(error, axis=0, keepdims=True)

    dA1 = np.dot(error, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

Z1 = np.dot(X_test, W1) + b1
A1 = relu(Z1)

Z2 = np.dot(A1, W2) + b2
A2 = softmax(Z2)

pred = np.argmax(A2, axis=1)
actual = np.argmax(Y_test, axis=1)

accuracy = np.mean(pred == actual)

print("Accuracy:", accuracy)