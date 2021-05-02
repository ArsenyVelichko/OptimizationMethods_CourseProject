from scipy.optimize import linprog as simplex_method
import numpy as np


class FeasibleDirectMethod:
    def __init__(self, dim):
        self.__dim = dim

        self.__main_func = lambda x: 0
        self.__main_grad = lambda x: np.zeros(dim)

        self.__bound_funcs = []
        self.__bound_grads = []

        self.__is_init_approx = False

        self.__steps = []
        self.__approx_steps = []

    def __active_bounds(self, x, delta):
        active_idxs = []

        for idx in range(len(self.__bound_funcs)):
            if -delta <= self.__bound_funcs[idx](x) <= 0.0:
                active_idxs.append(idx)

        return active_idxs

    def __solve_auxilary(self, x_k, d_k):
        bounds_idxs = self.__active_bounds(x_k, d_k)
        aux_matrix = np.zeros(shape=(1 + len(bounds_idxs), self.__dim + 1))
        aux_matrix[:, self.__dim] = -1
        aux_matrix[0, 0:self.__dim] = self.__main_grad(x_k)

        for i in range(len(bounds_idxs)):
            grad = self.__bound_grads[bounds_idxs[i]]
            aux_matrix[i + 1, 0:self.__dim] = grad(x_k)

        target_coefs = np.zeros(self.__dim + 1)
        target_coefs[self.__dim] = 1

        bias = np.zeros(aux_matrix.shape[0])
        bounds = [[-1, 1] for _ in range(self.__dim)]
        bounds.append([None, None])

        return simplex_method(c=target_coefs, A_ub=aux_matrix, b_ub=bias, bounds=bounds)

    def __define_step(self, x_k, eta_k, s_k):
        alpha = 1

        while True:
            diff = self.__main_func(x_k + alpha * s_k) - self.__main_func(x_k)
            first_eq = diff <= 0.5 * eta_k * alpha

            second_eq = True
            for func in self.__bound_funcs:
                val = func(x_k + alpha * s_k)
                second_eq = second_eq and val <= 0

            if first_eq and second_eq:
                return alpha

            alpha *= 0.5

    def __next_step(self, x_k, d_k):
        aux_result = self.__solve_auxilary(x_k, d_k)
        s_k = aux_result.x[0:self.__dim]

        if aux_result.fun < -d_k:
            alpha = self.__define_step(x_k, aux_result.fun, s_k)
            x_next = x_k + alpha * s_k
            d_next = d_k

        else:
            x_next = x_k
            d_next = 0.5 * d_k

        return x_next, d_next, aux_result.fun

    def __delta_0k(self, x_k, epsilon):
        values = []
        bounds_idxs = self.__active_bounds(x_k, epsilon)

        for idx in range(len(self.__bound_funcs)):
            if idx not in bounds_idxs:
                values.append(self.__bound_funcs[idx](x_k))

        return -max(values)

    def __wrap_bound_func(self, idx):
        return lambda x: self.__bound_funcs[idx](x) - x[self.__dim]

    def __wrap_bound_grad(self, idx):
        return lambda x: np.append(self.__bound_grads[idx](x), -1)

    def __create_approx_method(self):
        approx_method = FeasibleDirectMethod(self.__dim + 1)

        approx_func = lambda x: x[self.__dim]
        approx_grad = lambda x: np.append(np.zeros(self.__dim), 1)

        bound_funcs = []
        bound_grads = []
        for i in range(len(self.__bound_funcs)):
            bound_funcs.append(self.__wrap_bound_func(i))
            bound_grads.append(self.__wrap_bound_grad(i))

        approx_method.set_main_func(approx_func, approx_grad)
        approx_method.set_bound_funcs(bound_funcs, bound_grads)
        approx_method.__is_init_approx = True
        return approx_method

    def __max_bound_value(self, x):
        max_value = float('-inf')
        for func in self.__bound_funcs:
            max_value = max(max_value, func(x))

        return max_value

    def set_main_func(self, func, grad):
        self.__main_func = func
        self.__main_grad = grad

    def set_bound_funcs(self, funcs, grads):
        self.__bound_funcs = funcs
        self.__bound_grads = grads

    def solve(self, x_0, d_0, epsilon, max_iters=1000):
        self.__steps.clear()

        if not self.__is_init_approx:
            approx_method = self.__create_approx_method()

            eta_0 = self.__max_bound_value(x_0)
            x_0 = np.append(x_0, eta_0)

            init_approx = approx_method.solve(x_0, d_0, epsilon)
            x_0 = init_approx[0:self.__dim]
            d_0 = -init_approx[self.__dim]

            self.__approx_steps = approx_method.__steps

        x_k = x_0
        d_k = d_0

        for iters in range(max_iters):
            self.__steps.append(x_k)

            if self.__is_init_approx and self.__main_func(x_k) < 0:
                return x_k

            x_k, d_k, eta_k = self.__next_step(x_k, d_k)
            if d_k < self.__delta_0k(x_k, epsilon) and abs(eta_k) < epsilon:
                return x_k, iters

        return None

    def plot(self, axis):
        x = [point[0] for point in self.__approx_steps]
        y = [point[1] for point in self.__approx_steps]
        axis.plot(x, y, 'c--o')

        x = [point[0] for point in self.__steps]
        y = [point[1] for point in self.__steps]
        axis.plot(x, y, 'r--o')
