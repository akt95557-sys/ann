##ImplementArtificil Neural Network training process in Python by using Forward Propagation, Back Propagation
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

input_size = 2
hidden_size = 2
output_size = 1

np.random.seed(1)

weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

learning_rate = 0.1

for _ in range(10000):
    
    # Forward propagation
    hidden_layer = sigmoid(np.dot(X, weights_input_hidden))
    output_layer = sigmoid(np.dot(hidden_layer, weights_hidden_output))
    
    # Error
    error = y - output_layer
    
    # Backpropagation
    output_delta = error * sigmoid_derivative(output_layer)
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)
    
    # Update weights
    weights_hidden_output += hidden_layer.T.dot(output_delta) * learning_rate
    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

print("Final Output:")
print(output_layer)