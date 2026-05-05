#even odd
import numpy as np

def step(x):
    return 1 if x >= 0 else 0

training_data = [
    ([0,0,0,0,0,0], 1),
    ([0,0,0,0,0,1], 0),
    ([0,0,0,0,1,0], 1),
    ([0,0,0,0,1,1], 0),
    ([0,0,0,1,0,0], 1),
    ([0,0,0,1,0,1], 0),
    ([0,0,0,1,1,0], 1),
    ([0,0,0,1,1,1], 0),
    ([0,0,1,0,0,0], 1),
    ([0,0,1,0,0,1], 0)
]

weights = np.zeros(6)
learning_rate = 1

for data, label in training_data:
    x = np.array(data)
    y = label
    output = step(np.dot(x, weights))
    error = y - output
    weights += learning_rate * error * x

j = int(input("Enter a number (0-9): "))

binary_input = np.array([int(bit) for bit in format(j, '06b')])
prediction = step(np.dot(binary_input, weights))

if prediction == 1:
    print(j, "is Even")
else:
    print(j, "is Odd")