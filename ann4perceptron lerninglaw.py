#With a suitable example demonstrate the perceptron learning law with its decision regions using python. Give the output in graphical form.
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0], [1,0], [0,1], [1,1]])
Y = np.array([-1, -1, -1, 1])

w = np.zeros(2)
b = 0
lr = 0.3

for _ in range(10):
    for i in range(len(X)):
        y_pred = np.sign(np.dot(X[i], w) + b)
        
        if y_pred != Y[i]:
            w += lr * Y[i] * X[i]
            b += lr * Y[i]

x_min, x_max = -1, 2
y_min, y_max = -1, 2

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5)

plt.scatter(X[:, 0], X[:, 1], c='blue')

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Perceptron Decision Boundary")
plt.grid()
plt.show()