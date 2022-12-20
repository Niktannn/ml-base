import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


class SplitNode:
    def __init__(self, feature_ind, threshold, left, right):
        self.feature_ind = feature_ind
        self.threshold = threshold
        self.left = left
        self.right = right

    def do_predict(self, x):
        if x[self.feature_ind] < self.threshold:
            return self.left.do_predict(x)
        else:
            return self.right.do_predict(x)


class LeafNode:
    def __init__(self, answer):
        self.answer = answer

    def do_predict(self, _):
        return self.answer


def get_dt_predicts(points, tree):
    return [tree.do_predict(point) for point in points]


def build_dt(points, classes, order, weights, max_h, cur_h):
    n = len(order)
    cur_classes = classes[order].tolist()
    dominant_class = max(set(cur_classes), key=cur_classes.count)
    negative_rate = 0.
    for i in range(n):
        if classes[order[i]] != dominant_class:
            negative_rate += weights[order[i]]
    if cur_h == max_h or negative_rate == 0.:
        return LeafNode(dominant_class), negative_rate

    min_neg_rate = 1.
    best_left_tree = None
    best_right_tree = None
    best_feature_ind = 0
    best_threshold = 0.
    for feature_ind in range(len(points[0])):
        order.sort(key=lambda g: points[g][feature_ind])
        if points[order[0]][feature_ind] == points[order[n-1]][feature_ind]:
            continue
        for bound in range(n - 1):
            if points[order[bound]][feature_ind] == points[order[bound + 1]][feature_ind]:
                continue
            left_tree, left_neg_rate = build_dt(points, classes, order[:bound + 1], weights, max_h, cur_h + 1)
            right_tree, right_neg_rate = build_dt(points, classes, order[bound + 1:], weights, max_h, cur_h + 1)
            cur_neg_rate = left_neg_rate + right_neg_rate
            if cur_neg_rate < min_neg_rate:
                min_neg_rate = cur_neg_rate
                best_left_tree = left_tree
                best_right_tree = right_tree
                best_feature_ind = feature_ind
                best_threshold = (points[order[bound]][feature_ind] + points[order[bound + 1]][feature_ind]) / 2
    return SplitNode(best_feature_ind, best_threshold, best_left_tree, best_right_tree), min_neg_rate


def ada_boost_predict(point, trees, alphas):
    def sign(x):
        return 1 if x >= 0 else -1
    weighted_vote = 0
    for i in range(len(trees)):
        weighted_vote += alphas[i] * trees[i].do_predict(point)
    return sign(weighted_vote)


def get_ada_boost_predicts(points, trees, alphas):
    return [ada_boost_predict(point, trees, alphas) for point in points]


def calc_accuracy(predicted, classes):
    correctly_predicted = 0
    for i in range(len(classes)):
        correctly_predicted += int(classes[i] == predicted[i])
    return correctly_predicted / len(classes)


def process_data(dataset_name, max_h):
    points, classes = read_data(dataset_name)
    n = len(points)
    weights = [1/n] * n
    init_order = [i for i in range(n)]
    trees = []
    alphas = []

    iterations = 100
    steps_to_display = {1, 2, 3, 5, 8, 13, 21, 34, 55}
    accuracies = []

    for it in range(1, iterations + 1):
        tree, neg_rate = build_dt(points, classes, init_order, weights, max_h, 0)
        trees.append(tree)
        if neg_rate == 0:
            alpha = 0.5 * math.log((1-neg_rate + 1/n) / (neg_rate + 1/n))
        else:
            alpha = 0.5 * math.log((1-neg_rate) / neg_rate)
        alphas.append(alpha)
        cur_tree_predicts = get_dt_predicts(points, tree)
        weights_sum = 0.
        for j in range(n):
            weights[j] *= math.exp(-alpha * classes[j] * cur_tree_predicts[j])
            weights_sum += weights[j]
        for j in range(n):
            weights[j] /= weights_sum
        accuracies.append(calc_accuracy(get_ada_boost_predicts(points, trees, alphas), classes))
        if it in steps_to_display:
            draw("dataset: " + dataset_name + "; iteration: " + str(it)
                 + "; base algorithm: dt with max height=" + str(max_h),
                 points, classes, lambda dots: get_ada_boost_predicts(dots, trees, alphas))
    draw_acc_of_iteration("dataset: " + dataset_name + "; base algorithm: dt with max height=" + str(max_h),
                          accuracies, iterations)


def read_data(dataset_name):
    dataset = pd.read_csv(dataset_name + ".csv")
    points = np.array(dataset.values[:, :-1], dtype=float)
    class_to_num = np.vectorize(lambda cl: 1 if cl == 'P' else -1)
    classes = class_to_num(dataset.values[:, -1])
    return points, classes


def draw_acc_of_iteration(title, accuracies, max_step):
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.plot([i for i in range(1, max_step + 1)], accuracies)
    plt.show()


def draw(title, points, classes, get_predicts):
    x_min, y_min = np.amin(points, 0)
    x_max, y_max = np.amax(points, 0)
    step_x = (x_max - x_min) / 100
    step_y = (y_max - y_min) / 100
    x_min -= step_x
    x_max += step_x
    y_min -= step_y
    y_max += step_y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_x),
                         np.arange(y_min, y_max, step_y))

    mesh_dots = np.c_[xx.ravel(), yy.ravel()]
    zz = get_predicts(mesh_dots)
    zz = np.array(zz).reshape(xx.shape)

    plt.figure(figsize=(10, 10))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    x0, y0 = points[classes == -1].T
    x1, y1 = points[classes == 1].T

    plt.pcolormesh(xx, yy, zz, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(x0, y0, color='red', s=100)
    plt.scatter(x1, y1, color='blue', s=100)

    plt.title(title)
    plt.show()


max_height = 2
process_data("datasets/chips", max_height)
process_data("datasets/geyser", max_height)
