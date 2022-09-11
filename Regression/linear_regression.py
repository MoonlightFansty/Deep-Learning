# 线性回归
import numpy as np

# 计算loss
# loss = (wx+b-y) ** 2


def compute_error_for_line_given_points(w, b, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2  # 误差方差
    return totalError / float(len(points))  # 均方差

# 计算梯度信息
# w' = w - learning_rate * (Δloss / Δw)


def step_gradient(w_current, b_current, points, learning_rate):
    w_gradient = 0
    b_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        w_gradient += -(2 / N) * x * (y - (w_current * x + b_current))
        b_gradient += -(2 / N) * (y - (w_current * x + b_current))
    new_w = w_current - (learning_rate * w_gradient)
    new_b = b_current - (learning_rate * b_gradient)
    return [new_w, new_b]

# 迭代优化


def gradient_descent_runner(points, starting_w, starting_b,
                            learning_rate, num_iterations):
    w = starting_w
    b = starting_b
    for i in range(num_iterations):
        w, b = step_gradient(w, b, np.array(points), learning_rate)
    return [w, b]


def run():
    points = np.genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0001
    initial_w = 0  # 初始化w
    initial_b = 0  # 初始化b
    num_iterations = 1000
    print(f'Starting gradient descent at w = {initial_w}, b = {initial_b}, '
          f'error = {compute_error_for_line_given_points(initial_w, initial_b, points)}')
    print("Running...")
    [w, b] = gradient_descent_runner(points, initial_w, initial_b, learning_rate, num_iterations)
    print(f'After {num_iterations} iterations w = {w}, b = {b},'
          f'error = {compute_error_for_line_given_points(w, b, points)}')


if __name__ == '__main__':
    run()
