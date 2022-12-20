import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt


# distance functions
def minkowski_distance(row1, row2, p):
    distance = 0.0
    for i in range(len(row1)):
        distance += (abs(float(row1[i]) - float(row2[i]))) ** p
    return distance ** (1 / p)


def euclidean(row1, row2):
    return minkowski_distance(row1, row2, 2.0)


def manhattan(row1, row2):
    return minkowski_distance(row1, row2, 1.0)


def chebyshev(row1, row2):
    return max([abs(row1[i] - row2[i]) for i in range(len(row1))])


# kernels
def finite(arg, res):
    return res if (abs(arg) < 1) else 0


def uniform(x):
    return finite(x, 0.5)


def triangular(x):
    return finite(x, 1 - abs(x))


def epanechnikov(x):
    return finite(x, 0.75 * (1 - x ** 2))


def quartic(x):
    return finite(x, (float(15) / 16) * (1 - x ** 2) ** 2)


def triweight(x):
    return finite(x, (float(35) / 32) * (1 - x ** 2) ** 3)


def tricube(x):
    return finite(x, (float(70) / 81) * (1 - abs(x) ** 3) ** 3)


def cosine(x):
    return finite(x, (math.pi / 4) * math.cos((math.pi * x) / 2))


def gaussian(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-((x ** 2) / 2))


def logistic(x):
    return 1 / (math.exp(x) + math.exp(-x) + 2)


def sigmoid(x):
    return 2 / (math.pi * (math.exp(x) + math.exp(-x)))


def get_minmax(_dataset):
    res = list()
    for i in range(len(_dataset[0])):
        value_min = _dataset[:, i].min()
        value_max = _dataset[:, i].max()
        res.append([value_min, value_max])
    return res


def normalize(features, _minmax):
    for i in range(len(features)):
        for j in range(len(features[0])):
            features[i][j] = ((features[i][j] - _minmax[j][0]) / (_minmax[j][1] - _minmax[j][0]))


def get_neighbors_distances(train, target, is_fixed, dist_func, exclude):
    neighbours = list()
    for i in range(len(train)):
        if i == exclude:
            continue
        dist = dist_func(target, train[i])
        neighbours.append((i, dist))
    if not is_fixed:
        neighbours.sort(key=lambda t: t[1])
    return neighbours


def nadaraya_watson_reg(train, target, is_fixed, w, kernel, dist_fun, exclude):
    def get_mean_label(_neighbors, n):
        summ = np.zeros(n, dtype=float)
        for tup in _neighbors:
            summ += train[int(tup[0])][-n:]
        return summ / len(_neighbors)

    label_len = len(train[0]) - len(target)
    label_weighted_sum = np.zeros(label_len, dtype=float)
    weight_sum = 0.0

    neighbors = get_neighbors_distances(train, target, is_fixed, dist_fun, exclude)

    width = (w if is_fixed else neighbors[min(w, len(train) - 2)][1])
    if width == 0:
        matched_neighbors = np.array(list(filter(lambda x: (x[1] == 0), neighbors)))
        if len(matched_neighbors) > 0:
            return get_mean_label(matched_neighbors, label_len)
        else:
            return get_mean_label(neighbors, label_len)

    for neighbour in neighbors:
        weight = kernel(neighbour[1] / width)
        label = np.array(train[neighbour[0]][-label_len:], dtype=float)
        label_weighted_sum += label * weight
        weight_sum += weight

    if weight_sum == 0:
        return get_mean_label(neighbors, label_len)

    return label_weighted_sum / weight_sum


def nadaraya_watson_reg_h(train, test_row, h, kernel, dist_fun, exclude):
    return nadaraya_watson_reg(train, test_row, True, h, kernel, dist_fun, exclude)


def nadaraya_watson_reg_k(train, test_row, k, kernel, dist_fun, exclude):
    return nadaraya_watson_reg(train, test_row, False, k, kernel, dist_fun, exclude)


def convert_labels_naive(all_labels, _label_to_index):
    return [[_label_to_index[label]] for label in all_labels]


def convert_labels_one_hot(all_labels, _label_to_index):
    _converted_labels = []
    for label in all_labels:
        label_vector = np.zeros(len(_label_to_index), dtype=float)
        label_vector[_label_to_index[label]] += 1
        _converted_labels.append(label_vector)
    return _converted_labels


def make_dataset(_features, _labels):
    _dataset = []
    for i in range(len(_features)):
        row = _features[i].tolist()
        for label_val in _labels[i]:
            row.append(label_val)
        _dataset.append(row)
    return np.array(_dataset)


def draw_plot(dataset, feature_1_num, feature_2_num, target):
    class_to_color = {
        1: "r",
        2: "g",
        3: "b"
    }

    colored_data = list(map(lambda l: class_to_color[l], dataset[:, 0]))
    colored_data.append("y")

    x = list(dataset[:, feature_1_num])
    y = list(dataset[:, feature_2_num])

    x.append(target[feature_1_num])
    y.append(target[feature_2_num])

    plt.scatter(x, y, c=colored_data)
    plt.show()


def predict(reg_convert_method, converted_datasets, features_size, reg_method, width, kernel, metric, target_index):
    converted_dataset = converted_datasets[reg_convert_method]

    target = converted_dataset[target_index]
    target_features = target[:features_size]
    target_label = target[features_size:]

    predicted_label = list(
        reg_method(converted_dataset, target_features, width, kernel, metric, target_index))
    if len(predicted_label) == 1:
        predicted_index = int(round(predicted_label[0]))
        real_index = int(target_label[0])
    else:
        predicted_index = int(predicted_label.index(max(predicted_label)))
        real_index = target_label.tolist().index(max(target_label))
    return predicted_index, real_index


def f_b_score(precision, recall, b):
    if precision == 0 or recall == 0:
        return 0.0
    return (1 + b ** 2) * precision * recall / ((b ** 2) * precision + recall)


def get_f_score(confusion_matrix, labels_num, label_counts, dataset_size):
    recalls = np.array([confusion_matrix[i][i] for i in range(labels_num)], dtype=float)
    recall = sum(recalls) / dataset_size

    precisions = np.array([0 if sum(confusion_matrix[i]) == 0
                           else confusion_matrix[i][i] / sum(confusion_matrix[i])
                           for i in range(labels_num)], dtype=float)
    precision = np.average(precisions, weights=label_counts)
    return f_b_score(precision, recall, 1)


def print_parameters(parameters, min_loss, reg_convert_method):
    print("naive convertation" if reg_convert_method == convert_labels_naive else "one-hot convertation")
    print("best parameters:")
    for param in parameters[:-1]:
        print(param.__name__)
    if parameters[-1] == nadaraya_watson_reg_h:
        print("fixed window")
    else:
        print("variable window")
    print("average loss: ", min_loss)
    print("------------------------")


def main():
    dataset = pd.read_csv('bridges.csv')

    print(dataset)

    dataset_size = len(dataset)
    features_size = len(dataset.values[0]) - 1

    features = dataset.values[:, :-1]
    minmax = get_minmax(features)
    normalize(features, minmax)

    all_labels = dataset.values[:, -1]
    labels = list(set(all_labels))
    labels_number = len(labels)
    label_to_index = {labels[i]: i for i in range(len(labels))}

    reg_convert_methods = [convert_labels_naive, convert_labels_one_hot]
    converted_datasets = {m: make_dataset(features, m(all_labels, label_to_index)) for m in reg_convert_methods}

    print(converted_datasets[convert_labels_naive])
    print(converted_datasets[convert_labels_one_hot])

    metrics = [manhattan, euclidean, chebyshev]
    kernels = [uniform, triangular, epanechnikov, quartic, triweight, tricube, gaussian, cosine, logistic, sigmoid]

    min_data = np.zeros(features_size, dtype=float)
    max_data = np.ones(features_size, dtype=float)

    metric_window_widths = dict()
    for m in metrics:
        metric_window_widths[m] = (np.linspace(0, m(max_data, min_data), 10))

    neighbor_nums = [x for x in range(0, (dataset_size - 1), int(dataset_size / 10))]

    ans = []
    best_params = []

    for reg_convert_method in reg_convert_methods:
        min_loss = float(dataset_size)
        for metric in metrics:
            for kernel in kernels:
                for reg_method in [nadaraya_watson_reg_h, nadaraya_watson_reg_k]:
                    widths = metric_window_widths[metric] if reg_method == nadaraya_watson_reg_h else neighbor_nums
                    loss = 0.0
                    for width in widths:
                        for target_index in range(dataset_size):
                            predicted_index, real_index = predict(reg_convert_method, converted_datasets, features_size,
                                                                  reg_method, width, kernel, metric, target_index)
                            loss += (predicted_index != real_index)
                    loss_avg = loss / len(widths)
                    # print(loss_avg)
                    if loss_avg < min_loss:
                        min_loss = loss_avg
                        best_params = [metric, kernel, reg_method]
        ans.append(best_params)
        print_parameters(best_params, min_loss, reg_convert_method)

    for j in range(2):
        for reg_method in [nadaraya_watson_reg_h, nadaraya_watson_reg_k]:
            reg_convert_method = reg_convert_methods[j]
            metric = ans[j][0]
            kernel = ans[j][1]

            neighbor_nums = [x for x in range(0, dataset_size - 1, int(dataset_size / 100))]

            max_width = metric(max_data, min_data)
            metric_window_widths = (np.linspace(0, max_width, 100))

            widths = metric_window_widths if reg_method == nadaraya_watson_reg_h else neighbor_nums
            f_scores = list()

            for width in widths:
                confusion_matrix = np.zeros((labels_number, labels_number), dtype=float)
                label_counts = np.zeros(labels_number, dtype=float)
                for target_index in range(dataset_size):
                    predicted_index, real_index = predict(reg_convert_method, converted_datasets, features_size,
                                                          reg_method, width, kernel, metric, target_index)
                    label_counts[real_index] += 1
                    confusion_matrix[predicted_index][real_index] += 1
                f_scores.append(get_f_score(confusion_matrix, labels_number, label_counts, dataset_size))

            plt.plot(widths, f_scores)
            plt.xlabel('fixed width' if reg_method == nadaraya_watson_reg_h else 'number of neighbors')
            plt.title('naive convertation' if j == 0 else 'one-hot convertation')
            plt.show()


main()
