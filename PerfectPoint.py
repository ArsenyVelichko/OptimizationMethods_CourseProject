from FeasibleDirectMethod import FeasibleDirectMethod
import numpy as np
import matplotlib.pyplot as plt

# def main_func(x):
#     return 3 * x[0] + x[1] + 4 * np.sqrt(1 + 2 * x[0] ** 2 + 7 * x[1] ** 2)
#
#
# def main_grad(x):
#     return np.array([3 + 4 * 2 * x[0] / np.sqrt(1 + 2 * x[0] ** 2 + 7 * x[1] ** 2),
#                      1 + 4 * 7 * x[1] / np.sqrt(1 + 2 * x[0] ** 2 + 7 * x[1] ** 2)])


def euclid_dist(x):
    dx = x[0] - 7
    dy = x[1] - 10
    return dx * dx + dy * dy


def euclid_grad(x):
    dx = x[0] - 7
    dy = x[1] - 10
    return np.array([2 * dx, 2 * dy])



def bound_funcs():
    funcs = list()

    funcs.append(lambda x: -x[0] - x[1] + 8)
    funcs.append(lambda x: x[0] + x[1] - 16)
    funcs.append(lambda x: -x[0] + x[1] - 4)
    funcs.append(lambda x: x[0] - x[1])
    funcs.append(lambda x: 3 * x[0] - x[1] - 12)

    return funcs


def bound_grads():
    grads = list()

    grads.append(lambda x: np.array([-1, -1]))
    grads.append(lambda x: np.array([1, 1]))
    grads.append(lambda x: np.array([-1, 1]))
    grads.append(lambda x: np.array([1, -1]))
    grads.append(lambda x: np.array([3, -1]))

    return grads

def plot_solution_set(axis):
    funcs = [lambda x: -x + 8,
             lambda x: -x + 16,
             lambda x: x + 4,
             lambda x: x,
             lambda x: 3 * x - 12]
    intervals = [(2.0, 4.0), (6.0, 7.0), (2.0, 6.0), (4.0, 6.0), (6.0, 7.0)]

    for i in range(len(intervals)):
        x = np.linspace(intervals[i][0], intervals[i][1])
        y = funcs[i](x)
        axis.plot(x, y)


if __name__ == '__main__':
    x_0 = np.array([0, 0])
    d_0 = 1
    method = FeasibleDirectMethod(2)
    method.set_main_func(main_func, main_grad)
    method.set_bound_funcs(bound_funcs(), bound_grads())
    answer = method.solve(x_0, d_0, 10 ** -10)
    print(answer)

    fig, axis = plt.subplots()
    method.plot(axis)
    plot_solution_set(axis)

    fig.show()