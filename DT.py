import math
import random
import pandas as pd
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
# import time


def split_data(points, ind, b):
    points1 = list()
    points2 = list()
    for i in range(len(points)):
        if points[i][ind] < b:
            points1.append(points[i])
        else:
            points2.append(points[i])
    return points1, points2


def build_decision_tree(points, k, max_h, cur_h, cur_num, features_num, tree):
    n = len(points)
    m = len(points[0]) - 1
    if n == 1:
        tree[cur_num] = (True, points[0][-1])
        return

    init_counts = [0] * k
    for i in range(n):
        init_counts[points[i][-1]] += 1
    has_one_class = False
    max_class = 0
    for i in range(k):
        if init_counts[i] == n:
            has_one_class = True
            max_class = i
            break
        if init_counts[i] > init_counts[max_class]:
            max_class = i
    if has_one_class or cur_h == max_h:
        tree[cur_num] = (True, max_class)
        return

    # calculate init impurity
    init_imp = 0.
    for j in range(k):
        if init_counts[j] != 0:
            init_imp -= init_counts[j] * math.log2(init_counts[j])
    init_imp += n * math.log2(n)

    max_gain = 0.
    best_feat_ind = 0
    best_b = 0.
    indexes = [i for i in range(m)]
    if features_num != m:
        indexes = random.sample(indexes, features_num)
    for ind in indexes:
        points.sort(key=lambda point: point[ind])
        if points[0][ind] == points[n-1][ind]:
            continue
        counts = [0] * k
        imp1 = 0.
        imp2 = init_imp
        for bound in range(n - 1):
            cl = points[bound][-1]

            prev_count = counts[cl]
            if bound != 0:
                imp1 -= bound * math.log2(bound)
            imp1 += (bound + 1) * math.log2(bound + 1)
            if prev_count != 0:
                imp1 += prev_count * math.log2(prev_count)
            imp1 -= (prev_count + 1) * math.log2(prev_count + 1)

            prev_count = init_counts[cl] - counts[cl]
            imp2 -= (n - bound) * math.log2(n - bound)
            imp2 += (n - bound - 1) * math.log2(n - bound - 1)
            if prev_count != 0:
                imp2 += prev_count * math.log2(prev_count)
            if (prev_count - 1) != 0:
                imp2 -= (prev_count - 1) * math.log2(prev_count - 1)

            counts[cl] += 1
            if points[bound][ind] == points[bound + 1][ind]:
                # or points[bound + 1][ind] < cur_bound
                continue
            # cur_bound += step
            cur_imp = imp1 + imp2
            cur_gain = (init_imp - cur_imp) / n
            if cur_gain > max_gain:
                max_gain = cur_gain
                best_feat_ind = ind
                best_b = float(points[bound][ind] + points[bound + 1][ind]) / 2

    if max_gain == 0.:
        tree[cur_num] = (True, max_class)
        return

    points1, points2 = split_data(points, best_feat_ind, best_b)
    tree[cur_num] = (False, best_feat_ind, best_b)
    build_decision_tree(points1, k, max_h, cur_h + 1, 2 * cur_num + 1, features_num, tree)
    build_decision_tree(points2, k, max_h, cur_h + 1, 2 * cur_num + 2, features_num, tree)


def dt_classifier(point, tree):
    cur_num = 0
    while not tree[cur_num][0]:
        if point[tree[cur_num][1]] < tree[cur_num][2]:
            cur_num = cur_num * 2 + 1
        else:
            cur_num = cur_num * 2 + 2
    return tree[cur_num][1]


def get_predicts(points, tree):
    return [dt_classifier(point, tree) for point in points]


def get_best_h(train_points, test_points, k, max_h):
    m = len(train_points[0]) - 1
    max_accuracy = 0
    best_h = 0
    accuracies_train = []
    accuracies_test = []
    for h in range(max_h + 1):
        tree = [None for _ in range(int(2 ** (h + 1)))]
        build_decision_tree(train_points, k, h, 0, 0, m, tree)

        accuracy = calc_accuracy(get_predicts(test_points, tree), test_points)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_h = h

        accuracies_test.append(accuracy)
        accuracies_train.append(calc_accuracy(get_predicts(train_points, tree), train_points))

    return best_h, (accuracies_train, accuracies_test)


def build_random_forest(trees_num, train_points, test_points, k, max_h):
    n = len(train_points)
    m = len(train_points[0]) - 1
    all_predicts_test = [[] for _ in range(len(test_points))]
    all_predicts_train = [[] for _ in range(len(test_points))]
    for j in range(trees_num):
        points = random.choices(train_points, k=n)
        tree = [None for _ in range(int(2 ** (max_h + 1)))]
        build_decision_tree(points, k, max_h, 0, 0, int(math.sqrt(m)), tree)
        for g in range(len(test_points)):
            all_predicts_test[g].append(dt_classifier(test_points[g], tree))
        for g in range(len(train_points)):
            all_predicts_train[g].append(dt_classifier(train_points[g], tree))

    predicted_classes_train = [max(set(predicts), key=predicts.count) for predicts in all_predicts_train]
    predicted_classes_test = [max(set(predicts), key=predicts.count) for predicts in all_predicts_test]
    return calc_accuracy(predicted_classes_train, train_points), calc_accuracy(predicted_classes_test, test_points)


def calc_accuracy(predicted, points):
    correctly_predicted = 0
    for i in range(len(points)):
        correctly_predicted += int(points[i][-1] == predicted[i])
    return correctly_predicted / len(points)


def main():
    dirname = "datasets"
    files = [join(dirname, f) for f in listdir(dirname) if isfile(join(dirname, f))]
    trees_num = 50
    min_optimal_h = 100
    max_optimal_h = 0
    height_min = 0
    height_max = 0
    accuracies_min = None
    accuracies_max = None
    for i in range(0, len(files), 2):
        print("dataset №", i // 2 + 1)
        test_points = pd.read_csv(files[i]).values.tolist()
        train_points = pd.read_csv(files[i+1]).values.tolist()
        k = 0
        for j in range(len(train_points)):
            if train_points[j][-1] > k:
                k = train_points[j][-1]
            train_points[j][-1] -= 1
        for j in range(len(test_points)):
            test_points[j][-1] -= 1
        height = int(math.log2(len(train_points) + 1))
        best_h, accuracies = get_best_h(train_points, test_points, k, height)
        print("optimal height for decision tree:", best_h)

        if best_h < min_optimal_h:
            accuracies_min = accuracies
            height_min = height
            min_optimal_h = best_h
        if best_h > max_optimal_h:
            accuracies_max = accuracies
            height_max = height
            max_optimal_h = best_h

        accuracy_train, accuracy_test = build_random_forest(trees_num, train_points, test_points, k, height)
        print("accuracy for random forest on train:", accuracy_train)
        print("accuracy for random forest on test:", accuracy_test)
        print("------------")

    draw_accuracy_of_height(height_min, accuracies_min, "min")
    draw_accuracy_of_height(height_max, accuracies_max, "max")


def draw_accuracy_of_height(height, accuracies, mode):
    plt.xlabel("max height")
    plt.ylabel("accuracy")
    plt.plot([i for i in range(height + 1)], accuracies[0], color='b', label='train', linestyle='-', marker='.')
    plt.plot([i for i in range(height + 1)], accuracies[1], color='r', label='test', linestyle='-', marker='.')
    plt.title("for dataset with " + mode + " optimal height")
    plt.legend()
    plt.show()


main()
