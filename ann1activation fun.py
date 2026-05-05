#Write a Python program to plot a few activation functions that are being used in neural networks
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.plot(x, relu(x), label='ReLU', linewidth=3, linestyle='--')
plt.plot(x, linear(x), label='Linear')

# Softmax (constant-style visualization for plotting purpose)
plt.plot(x, softmax(x), label='Softmax (approx)', linestyle='dotted')

plt.xlabel("X values")
plt.ylabel("Activation Output")
plt.legend()
plt.title("Activation Functions")
plt.grid()
plt.show()
