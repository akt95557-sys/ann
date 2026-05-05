#Generate ANDNOT function using McCulloch-Pitts neural net by a python program.import numpy as np

def mp_neuron(inputs, weights, threshold):
    weighted_sum = np.dot(inputs, weights)
    return 1 if weighted_sum >= threshold else 0

def and_not(x1, x2):
    weights = np.array([1, -1])
    threshold = 1
    inputs = np.array([x1, x2])
    return mp_neuron(inputs, weights, threshold)

print("x1 x2 -> Output")

for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(x1, x2, "->", and_not(x1, x2))