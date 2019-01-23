import numpy as np
import matplotlib.pyplot as plt


def draw(x1, x2):
    ln = plt.plot(x1, x2)
    plt.pause(0.0001)
    ln[0].remove()


def sigmoid(score):
    return (1 / (1 + np.exp(-score)))


def calc_error(points, line_parameters, y):
    m = points.shape[0]
    p = sigmoid(points * line_parameters)
    cross_entropy = -(np.log(p).T * y + np.log(1 - p).T * (1 - y)) / m
    return cross_entropy


def grad_descent(points, y, alpha, line_parameters):
    m = points.shape[0]
    for i in range(2000):
        p = sigmoid(points * line_parameters)
        grad = (points.T * (p - y)) * (alpha / m)
        line_parameters = line_parameters - grad
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -(w1 / w2) * x1 - (b / w2)
        draw(x1, x2)
        print(calc_error(points, line_parameters,y))
    plt.plot(x1,x2)


n_pts = 100
np.random.seed(0)
bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
bot_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
all_points = np.vstack((top_region, bot_region))
line_parameters = np.matrix([np.zeros(3)]).T
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(2 * n_pts, 1)
print(y.shape)

_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bot_region[:, 0], bot_region[:, 1], color='b')
grad_descent(all_points, y, 0.06, line_parameters)
plt.show()

print(calc_error(all_points, line_parameters, y))
