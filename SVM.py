import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt


def linear_kernel(a, b):
    return np.dot(a, b)


def gen_linear_params():
    yield [None]


def polynomial_kernel(a, b, gamma=1., k0=1., degree=3.):
    return np.power(gamma * np.dot(a, b) + k0, degree)


def gen_polynomial_params():
    for gamma in [0.01, 0.5, 1.0]:
        for k0 in [0.0, 1.0]:
            for degree in range(2, 6):
                yield [gamma, k0, float(degree)]


def gaussian_kernel(a, b, gamma=0.5):
    return np.exp(-gamma * np.power(np.linalg.norm(a - b), 2))


def gen_gaussian_params():
    for gamma in [0.1, 0.3, 0.5, 1., 2., 3., 4., 5.]:
        yield [gamma]


def get_kernel_func(name, params):
    if name == "polynomial":
        return lambda a, b: polynomial_kernel(a, b, *params)
    elif name == "gaussian":
        return lambda a, b: gaussian_kernel(a, b, *params)
    else:
        return lambda a, b: linear_kernel(a, b)


def smo(points, classes, c, kernel, eps, max_iteration):
    def is_non_bound(a):
        return a > eps and a + eps < c

    def calc_class(ind):
        return (alphas * classes).dot(kernel[ind]) - w0

    def calc_alpha_bounds(y1, y2, a1, a2):
        if y1 != y2:
            return max(0., a2 - a1), min(c, c + a2 - a1)
        else:
            return max(0., a2 + a1 - c), min(c, a2 + a1)

    def calc_error(ind):
        if is_non_bound(alphas[ind]):
            return errors[ind]
        else:
            return calc_class(ind) - classes[ind]

    def examine_example(i2):
        def take_step():
            def calc_w0():
                w0_1 = w0 + (e1 + y1 * (a1 - alpha_1) * k11 + y2 * (a2 - alpha_2) * k12)
                w0_2 = w0 + (e2 + y1 * (a1 - alpha_1) * k12 + y2 * (a2 - alpha_2) * k22)
                if is_non_bound(a1):
                    return w0_1
                if is_non_bound(a2):
                    return w0_2
                return (w0_1 + w0_2) / 2

            def calc_f_at_bounds():
                f1 = y1 * (e1 + w0) - alpha_1 * k11 - s * alpha_2 * k12
                f2 = y2 * (e2 + w0) - s * alpha_1 * k12 - alpha_2 * k22
                l1 = alpha_1 + s * (alpha_2 - l)
                h1 = alpha_1 + s * (alpha_2 - h)
                return (l1 * f1 + l * f2 + 0.5 * (l1 ** 2) * k11 + 0.5 * (l ** 2) * k22 + s * l * l1 * k12,
                        h1 * f1 + h * f2 + 0.5 * (h1 ** 2) * k11 + 0.5 * (h ** 2) * k22 + s * h * h1 * k12)

            def update_errors(t2, t1, w0_delta):
                for k in range(n):
                    if is_non_bound(alphas[k]):
                        errors[k] += t2 * kernel[i2][k] + t1 * kernel[i1][k] - w0_delta
                errors[i1] = 0.
                errors[i2] = 0.

            alpha_1 = alphas[i1]
            y1 = classes[i1]
            l, h = calc_alpha_bounds(y1, y2, alpha_1, alpha_2)
            if h - l < eps:
                return False
            e1 = calc_error(i1)
            s = float(y1 * y2)
            k11 = kernel[i1][i1]
            k12 = kernel[i1][i2]
            k22 = kernel[i2][i2]
            eta = k11 + k22 - 2 * k12
            if eta > 0.:
                a2 = alpha_2 + y2 * (e1 - e2) / eta
                if a2 < l:
                    a2 = l
                if a2 > h:
                    a2 = h
            # under unusual circumstances, e g bad kernel(eta>0) or same points(eta=0)
            else:
                f_l, f_h = calc_f_at_bounds()
                if f_l < f_h - eps:
                    a2 = l
                elif f_l > f_h + eps:
                    a2 = h
                else:
                    a2 = alpha_2
            # no progress
            if abs(a2 - alpha_2) < eps * (a2 + alpha_2 + eps):
                return False
            a1 = alpha_1 + s * (alpha_2 - a2)
            nonlocal w0
            new_w0 = calc_w0()
            update_errors(y2 * (a2 - alpha_2), y1 * (a1 - alpha_1), new_w0 - w0)
            w0 = new_w0
            alphas[i1] = a1
            alphas[i2] = a2
            return True

        y2 = classes[i2]
        alpha_2 = alphas[i2]
        e2 = calc_error(i2)
        r2 = e2 * y2
        i1 = -1
        if (r2 < -eps and alpha_2 < c) or (r2 > eps and alpha_2 > 0):
            max_dif = 0.
            for j in range(n):
                if j != i2 and is_non_bound(alphas[j]):
                    error_dif = abs(e2 - errors[j])
                    if error_dif > max_dif:
                        max_dif = error_dif
                        i1 = j
            if i1 != -1 and take_step():
                return 1
            # under unusual circumstances, e g same points
            order = np.arange(n)
            np.random.shuffle(order)
            checked = set()
            if i1 != -1:
                for j in range(n):
                    i1 = order[j]
                    if i1 != i2 and is_non_bound(alphas[i1]):
                        if take_step():
                            return 1
                        else:
                            checked.add(i1)
            for j in range(n):
                i1 = order[j]
                if i1 != i2 and not (i1 in checked):
                    if take_step():
                        return 1
        return 0

    n = len(points)
    alphas = np.zeros(n, dtype=float)
    errors = np.array(-classes, dtype=float)
    w0 = 0.
    num_changed = 0
    examine_all = True
    iteration = 0
    while (num_changed > 0 or examine_all) and iteration < max_iteration:
        iteration += 1
        num_changed = 0
        if examine_all:
            for i in range(n):
                num_changed += examine_example(i)
        else:
            for i in range(n):
                if is_non_bound(alphas[i]):
                    num_changed += examine_example(i)
        if examine_all:
            examine_all = False
        elif num_changed == 0:
            examine_all = True
    return alphas, w0


def calc_kernel_matrix(points, kernel):
    n = len(points)
    kernel_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i <= j:
                kernel_matrix[i][j] = kernel(points[i], points[j])
            else:
                kernel_matrix[i][j] = kernel_matrix[j][i]
    return kernel_matrix


def sign(x):
    return 1 if x >= 0. else -1


def svm(target_points, train_points, train_classes, kernel_fun, c):
    eps = 0.0001
    max_iteration = 100
    kernel_matrix = calc_kernel_matrix(train_points, kernel_fun)
    alphas, b = smo(train_points, train_classes, c, kernel_matrix, eps, max_iteration)
    # get indices of support vectors
    supports = list()
    # for i in range(len(alphas)):
    #     if alphas[i] > eps and alphas[i] + eps < c:
    #         supports.append(i)
    # predict classes for target points
    kernel_products = [np.array([kernel_fun(point, target) for point in train_points]) for target in target_points]
    y_alpha = alphas * train_classes
    predicts = [sign(y_alpha.dot(products) - b) for products in kernel_products]
    return {"predicts": predicts, "supports": supports}


def draw(dataset, points, classes, kernel_name, kernel_params, c):
    kernel_fun = get_kernel_func(kernel_name, kernel_params)
    order = np.arange(len(points))
    np.random.shuffle(order)
    shuffled_points = points[order]
    shuffled_classes = classes[order]

    x_min, y_min = np.amin(points, 0)
    x_max, y_max = np.amax(points, 0)
    step_x = (x_max - x_min) / 100
    step_y = (y_max - y_min) / 500
    x_min -= step_x
    x_max += step_x
    y_min -= step_y
    y_max += step_y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_x),
                         np.arange(y_min, y_max, step_y))

    mesh_dots = np.c_[xx.ravel(), yy.ravel()]
    clf = svm(mesh_dots, shuffled_points, shuffled_classes, kernel_fun, c)
    clf2 = svm(shuffled_points, shuffled_points, shuffled_classes, kernel_fun, c)

    correctly_predicted = 0
    for i in range(len(shuffled_classes)):
        if shuffled_classes[i] == clf2["predicts"][i]:
            correctly_predicted += 1

    accuracy = correctly_predicted / len(classes)

    zz = clf["predicts"]
    zz = np.array(zz).reshape(xx.shape)

    plt.figure(figsize=(10, 10))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    x0, y0 = points[classes == -1].T
    x1, y1 = points[classes == 1].T

    plt.pcolormesh(xx, yy, zz, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(x0, y0, color='red', s=100)
    plt.scatter(x1, y1, color='blue', s=100)

    # sup_ind = clf["supports"]
    # support_points = points[sup_ind]
    # x_sup, y_sup = support_points.T

    # plt.scatter(x_sup, y_sup, color='white', marker='x', s=60)
    plt.title(dataset + ": " + kernel_name
              + ("" if kernel_params == [None] else " with parameters " + str(kernel_params))
              + " ; accuracy = " + str(accuracy))
    plt.show()


def k_fold(k, points, classes):
    positive_class_points = points[classes == 1]
    negative_class_points = points[classes == -1]

    for i in range(k):
        points_train_block = list()
        classes_train_block = list()
        points_test_block = list()
        classes_test_block = list()
        for j in range(len(positive_class_points)):
            if j % k == i:
                points_test_block.append(positive_class_points[j])
                classes_test_block.append(1)
            else:
                points_train_block.append(positive_class_points[j])
                classes_train_block.append(1)
        for j in range(len(negative_class_points)):
            if j % k == i:
                points_test_block.append(negative_class_points[j])
                classes_test_block.append(-1)
            else:
                points_train_block.append(negative_class_points[j])
                classes_train_block.append(-1)
        train_indices = np.arange(len(points_train_block))
        np.random.shuffle(train_indices)
        yield (np.array(points_train_block)[train_indices],
               np.array(classes_train_block)[train_indices],
               np.array(points_test_block),
               np.array(classes_test_block))


def process_dataset(dataset_name):
    dataset = pd.read_csv(dataset_name + '.csv')
    points = np.array(dataset.values[:, :-1], dtype=float)
    class_to_num = np.vectorize(lambda cl: 1 if cl == 'P' else -1)
    classes = class_to_num(dataset.values[:, -1])

    kernels = {"linear": linear_kernel, "polynomial": polynomial_kernel, "gaussian": gaussian_kernel}
    params_generators = {"linear": gen_linear_params, "polynomial": gen_polynomial_params,
                         "gaussian": gen_gaussian_params}
    cs = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    for kernel in kernels:
        max_accuracy = 0.
        best_kernel_params = None
        best_c = None
        for params in params_generators[kernel]():
            kernel_fun = get_kernel_func(kernel, params)
            for c in cs:
                print(kernel, params, c)
                correctly_predicted = 0
                total_count = 0
                for train_points, train_classes, test_points, test_classes in k_fold(6, points, classes):
                    predicted_classes = svm(test_points, train_points, train_classes, kernel_fun, c)["predicts"]
                    for i in range(len(test_classes)):
                        if predicted_classes[i] == test_classes[i]:
                            correctly_predicted += 1
                    total_count += len(test_classes)
                accuracy = float(correctly_predicted) / total_count
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    best_kernel_params = params
                    best_c = c
        print("kernel: " + kernel + "; accuracy: " + str(max_accuracy) +
              "; best parameters: " + str(best_kernel_params) + "; best C: " + str(best_c))
        draw(dataset_name, points, classes, kernel, best_kernel_params, best_c)


for ds_name in ["chips", "geyser"]:
    process_dataset(ds_name)