import numpy as np
import random
from matplotlib import pyplot as plt


def rmse(weights, features, labels, b=0.):
    return np.sqrt((((np.dot(features, weights) + b) - labels) ** 2).mean())


def nrmse(weights, features, labels, scatter, b=0.):
    if scatter == 0:
        scatter = np.inf
    return rmse(weights, features, labels, b) / scatter


def gen_grad_desc_params(dataset_size):
    left = min(10, dataset_size)
    right = min(50, dataset_size) + 1
    step = 10
    for b in range(left, right, step):
        for reg_factor in np.linspace(0., .75, 4):
            yield [b, reg_factor]


def grad_desc(b, reg_factor, max_iteration, features, labels, has_plot=False):
    n = len(features[0])
    scatter = np.amax(labels) - np.amin(labels)
    eps = 0.001
    min_loss = np.inf
    ans_weight = np.zeros(n, dtype=float)
    # начальные приближения
    w_inits = list()
    w_inits.append(np.random.uniform(-1. / (2 * n), 1. / (2 * n), size=n))
    w_inits.append((np.transpose(features).dot(labels)) / np.sum(features ** 2, axis=0))
    w_inits.append(np.zeros(n, dtype=float))
    # ищем лучшую точку(прибл)
    smooth_rate = 0.5
    best_losses = list()
    for w_init in w_inits:
        weight, losses = mini_batch_gradient_descent(b, w_init, eps, reg_factor, smooth_rate, features,
                                                                     labels, max_iteration)
        loss = nrmse(weight, features, labels, scatter)
        if loss < eps:
            ans_weight = weight
            best_losses = losses
            break
        if loss < min_loss:
            best_losses = losses
            min_loss = loss
            ans_weight = weight
    if has_plot:
        plt.plot([i for i in range(len(best_losses))], best_losses)
        plt.xlabel('iteration')
        plt.ylabel('NRMSE')
        plt.title('dataset ' + str(ds_num) + ': mini-batch gradient descent')
        plt.show()
    return ans_weight


def mini_batch_gradient_descent(batch_size, w_0, eps, reg_factor, smooth_rate, features, labels, max_iteration):
    n = len(features)
    w = w_0
    losses = list()
    loss = 0.
    prev_loss = 0.0
    order = np.arange(n)
    const_steps_passed = 0
    const_steps_to_stop = 100
    iteration = 0
    left = 0
    while iteration < max_iteration and const_steps_passed < const_steps_to_stop:
        if left >= n:
            left = 0
            np.random.shuffle(order)
        right = min(n, left + batch_size)

        # выделение батча
        features_batch = np.array([features[order[i]] for i in range(left, right)], dtype=float)
        labels_batch = np.array([labels[order[i]] for i in range(left, right)], dtype=float)
        scatter = np.amax(labels_batch) - np.amin(labels_batch)
        # считаем градиент
        deltas = np.dot(features_batch, w) - labels_batch
        grad = np.dot(np.transpose(features_batch), deltas) * 2.
        # шаг
        new_labels = features_batch.dot(grad - w * reg_factor)
        new_labels_square_sum = (new_labels ** 2).sum()
        if new_labels_square_sum == 0:
            conv_rate = 0
        else:
            conv_rate = deltas.dot(new_labels) / new_labels_square_sum
        # применяем шаг
        w = w * (1 - conv_rate * reg_factor) - conv_rate * grad

        # скользящее среднее
        if len(losses) == 0:
            loss = np.sqrt((deltas ** 2).mean()) / (1. if scatter == 0. else scatter)
            losses.append(loss)
        else:
            loss = (1 - smooth_rate) * loss + smooth_rate * np.sqrt((deltas ** 2).mean()) / (
                1. if scatter == 0. else scatter)
            losses.append(loss)
        # проверка останова
        if abs(loss - prev_loss) < eps:
            const_steps_passed += 1
        else:
            const_steps_passed = 0
        prev_loss = loss
        iteration += 1
        left = right
    return w, losses


def least_squares_params():
    for reg_factor in np.linspace(0., .5, 6):
        yield [reg_factor]


def least_squares_svd(reg_factor, max_iteration, features, labels, ):
    n = len(features[0])
    v, s, ut = np.linalg.svd(features, full_matrices=False)
    weights = np.zeros(n, dtype=float)
    for i in range(len(s)):
        coefficient = s[i] / (s[i] ** 2 + reg_factor)
        weights += coefficient * np.inner(ut[i], np.inner(v[:, i], labels))
    return weights


def genetic_params():
    eps = 0.1
    for pop_size in [10, 40, 100]:
        for elite_size in range(5, int(pop_size / 2) + 1, int(pop_size / 4)):
            for mut_rate in [0., 0.2, 0.4, 0.6]:
                yield [pop_size, elite_size, mut_rate, eps]


def genetic(init_pop_size, elite_size, mutation_rate, eps, max_iteration, features, labels):
    def crossing_over(x1, x2):
        for i in range(len(x1)):
            if random.getrandbits(1):
                x1[i] = x2[i]

    def mutation(x):
        av = x.mean()
        for i in range(int(len(x) * mutation_rate)):
            x[random.randint(0, len(x) - 1)] += random.uniform(-av, av)

    genes_num = len(features[0])
    max_ratio = max([labels[i] / features[i].sum() for i in range(len(features))])
    init_population = list()
    for i in range(int(init_pop_size / 2)):
        init_population.append(np.random.uniform(-1. / (2 * genes_num), 1. / (2 * genes_num), size=genes_num))
    for i in range(1, int(init_pop_size / 2)):
        init_population.append(
            np.random.uniform(-2. * max_ratio * i / init_pop_size, 2. * max_ratio * i / init_pop_size,
                              size=genes_num))
    init_population.append((np.transpose(features).dot(labels)) / np.sum(features ** 2, axis=0))
    init_population.append(np.zeros(genes_num, dtype=float))

    population = [[weight, ((features.dot(weight) - labels) ** 2).sum()] for weight in init_population]
    population.sort(key=lambda t: t[1])

    for k in range(max_iteration):
        if np.sqrt(population[0][1] / len(features)) < eps:
            return population[0][0]
        for i in range(elite_size, len(population)):
            crossing_over(population[i][0], population[random.randint(0, elite_size - 1)][0])
            mutation(population[i][0])
            population[i][1] = ((features.dot(population[i][0]) - labels) ** 2).sum()
        population.sort(key=lambda t: t[1])

    return population[0][0]


def k_fold_cross_validation(k, features, features_sorted, labels):
    size = len(features)
    for i in range(k):
        features_test_block = list()
        features_train_block = list()
        labels_train_block = list()
        labels_test_block = list()
        for j in range(0, size):
            if j % k == i:
                index = features_sorted[j][0]
                features_test_block.append(features[index])
                labels_test_block.append(labels[index])
            else:
                index = features_sorted[j][0]
                features_train_block.append(features[index])
                labels_train_block.append(labels[index])
        yield (np.array(features_test_block),
               np.array(features_train_block),
               np.array(labels_train_block),
               np.array(labels_test_block))


def z_normalize(matrix):
    means = matrix.mean(axis=0)
    stds = np.std(matrix, axis=0)
    return ((matrix - means) / stds), means, stds


def read_dataset(n):
    features = list()
    labels = list()
    for i in range(n):
        row = list(map(float, fin.readline().split()))
        features.append(row[:-1])
        labels.append(row[-1])
    features = np.array(features)
    features = features[:, ~np.all(features[1:] == features[:-1], axis=0)]
    labels = np.array(labels)
    return features, labels


def process_data():
    fin.readline()
    init_train_features, init_train_labels = read_dataset(int(fin.readline()))
    train_size = len(init_train_features)
    train_features, features_means, features_stds = z_normalize(init_train_features)
    train_labels, labels_mean, labels_std = z_normalize(init_train_labels)

    test_size = int(fin.readline())
    test_features, test_labels = read_dataset(test_size)

    features_sorted = [(i, train_labels[i]) for i in range(len(train_features))]
    features_sorted.sort(key=lambda t: t[1])

    blocks_num = min(5, max(int(train_size / 10), 2))
    max_iteration = 1000

    methods = [grad_desc, least_squares_svd, genetic]
    methods_params = [gen_grad_desc_params(train_size - int(train_size / blocks_num)),
                      least_squares_params(),
                      genetic_params()]
    methods_names = ["mini-batch gradient descent", "least squares with svd", "Holland genetic algorithm"]
    tuned_params = []

    for i in range(len(methods)):
        # if i == 2:
        #     continue
        min_rmse_sum = np.inf
        rmse_sum = 0.
        best_params = []
        for params in methods_params[i]:
            print(params)
            for (feat_test_block, feat_train_block, lab_train_block, lab_test_block) in \
                    k_fold_cross_validation(blocks_num, train_features, features_sorted, train_labels):
                weight = methods[i](*params, max_iteration, feat_train_block, lab_train_block)
                rmse_sum += rmse(weight, feat_test_block, lab_test_block)
            if rmse_sum < min_rmse_sum:
                min_rmse_sum = rmse_sum
                best_params = params
        print("best params for", methods_names[i], ":", best_params)
        tuned_params.append(best_params)

    scatter = np.amax(init_train_labels) - np.amin(init_train_labels)
    max_iterations = [int(10 ** i) for i in range(1, 6)]

    for i in range(len(methods)):
        # if i == 2:
        #     continue
        nrms_errors_train = list()
        nrms_errors_test = list()
        method = methods[i]
        if method == least_squares_svd:
            weight = method(*tuned_params[i], 0, train_features, train_labels)
            back_normalized_weight = (weight / features_stds) * labels_std
            b = labels_mean - np.inner(back_normalized_weight, features_means)
            nrmse_train = nrmse(back_normalized_weight, init_train_features, init_train_labels, scatter, b)
            nrms_errors_train = [nrmse_train] * len(max_iterations)
            nrmse_test = nrmse(back_normalized_weight, test_features, test_labels, scatter, b)
            nrms_errors_test = [nrmse_test] * len(max_iterations)
        else:
            if method == grad_desc:
                method(*tuned_params[i], 3000, train_features, train_labels, True)
            for max_iteration in max_iterations:
                weight = method(*tuned_params[i], max_iteration, train_features, train_labels)
                back_normalized_weight = (weight / features_stds) * labels_std
                b = labels_mean - np.inner(back_normalized_weight, features_means)
                nrmse_train = nrmse(back_normalized_weight, init_train_features, init_train_labels, scatter, b)
                nrms_errors_train.append(nrmse_train)
                nrmse_test = nrmse(back_normalized_weight, test_features, test_labels, scatter, b)
                nrms_errors_test.append(nrmse_test)

        plt.loglog(max_iterations, nrms_errors_train, label='train')
        plt.loglog(max_iterations, nrms_errors_test, label='test')
        plt.xlabel('number of iterations')
        plt.ylabel('NRMSE')
        plt.title("dataset " + str(ds_num) + ": " + methods_names[i])
        plt.legend()
        plt.show()


for ds_num in range(1, 8):
    fin = open(str(ds_num) + ".txt", "r")
    print("dataset " + str(ds_num))
    process_data()
