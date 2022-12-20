import math
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join


def read_file(file):
    f = open(file, 'r')
    subject = list(map(int, f.readline().split()[1:]))
    f.readline()
    body = list(map(int, f.readline().split()))
    words = subject + body
    is_spam = "spmsg" in file
    f.close()
    return words, int(is_spam)


def read_part(num):
    dirname = "datasets/part" + str(num)
    files = [join(dirname, f) for f in listdir(dirname) if isfile(join(dirname, f))]
    messages = []
    classes = []
    for file in files:
        words, cl = read_file(file)
        messages.append(words)
        classes.append(cl)
    return messages, classes


def read_all_parts():
    parts_mes = []
    parts_cl = []
    for i in range(1, 11):
        messages, classes = read_part(i)
        parts_mes.append(messages)
        parts_cl.append(classes)
    return parts_mes, parts_cl


def k_fold(k):
    for i in range(k):
        train_msgs = []
        train_cls = []
        test_msgs = []
        test_cls = []
        for j in range(k):
            if j != i:
                train_msgs.extend(parts_messages[j])
                train_cls.extend(parts_classes[j])
            else:
                test_msgs.extend(parts_messages[j])
                test_cls.extend(parts_classes[j])
        yield train_msgs, train_cls, test_msgs, test_cls


def get_counts(n, messages, classes):
    classes_counts = [0, 0]
    ngrams_counts = [{} for _ in range(n)]
    for i in range(len(messages)):
        cl = classes[i]
        classes_counts[cl] += 1
        for k in range(n):
            ngrams = set([tuple(messages[i][j:j+(k + 1)]) for j in range(len(messages[i]) - k)])
            for ngram in ngrams:
                ngrams_counts[k].setdefault(ngram, [0]*2)
                ngrams_counts[k][ngram][cl] += 1
    return classes_counts, ngrams_counts


def naive_bayes(n=1, alpha=1., loss_legit=1., loss_spam=1.):
    losses = [loss_legit, loss_spam]
    TP = 0
    TN = 0
    P = 0
    N = 0
    All = 0
    FP = 0
    for k in range(10):
        # print("tests part " + str(k))
        classes_counts = classes_counts_list[k]
        ngrams_counts = ngrams_counts_list[k]
        target_messages = parts_messages[k]
        target_classes = parts_classes[k]

        init_scores = [math.log(losses[i] * classes_counts[i]) for i in range(2)]
        freq_lns = {}
        for words_tup in ngrams_counts[n-1].keys():
            cl_freq_lns = []
            for i in range(2):
                count = ngrams_counts[n-1][words_tup][i]
                all_count = classes_counts[i]
                # if n == 1 else ngrams_counts[n-2][words_tup[:-1]][i]
                all_count_ln = math.log(all_count + 2 * alpha)
                cl_freq_lns.append(math.log(count + alpha) - math.log(all_count - count + alpha))
                init_scores[i] += (math.log(all_count - count + alpha) - all_count_ln)
            freq_lns[words_tup] = cl_freq_lns

        predicted_classes = []

        for i in range(len(target_messages)):
            scores = init_scores.copy()
            ngrams = set([tuple(target_messages[i][j:j+n]) for j in range(len(target_messages[i]) - n + 1)])
            for ngram in ngrams:
                if ngram in ngrams_counts[n-1].keys():
                    for j in range(2):
                        scores[j] += freq_lns[ngram][j]
            # print(scores)
            predicted_classes.append(int(scores[0] < scores[1]))

        All += len(target_classes)
        for i in range(len(target_classes)):
            if target_classes[i]:
                P += 1
                TP += int(target_classes[i] == predicted_classes[i])
            else:
                N += 1
                TN += int(target_classes[i] == predicted_classes[i])
                FP += int(target_classes[i] != predicted_classes[i])

    accuracy = float(TP + TN) / All
    recall = 0 if P == 0 else float(TP) / P
    specificity = 0 if N == 0 else float(TN) / N
    return accuracy, recall, specificity, FP


def draw_roc(n, alpha):
    xs = []
    ys = []
    for k in range(-300, 301, 20):
        _, recall, specificity, _ = naive_bayes(n, alpha, 10 ** k)
        xs.append(1 - specificity)
        ys.append(recall)
    plt.plot(xs, ys, linestyle='-', marker='.')
    plt.title("ROC")
    plt.xlabel("1-Specificity")
    plt.ylabel("Recall")
    plt.show()


def draw_accuracy_of_loss(n, alpha, max_legit_loss_pow):
    xs = []
    ys = []
    for k in range(0, max_legit_loss_pow + 1, 5):
        accuracy, _, _, _ = naive_bayes(n, alpha, 10 ** k)
        xs.append(10**k)
        ys.append(accuracy)
    plt.semilogx(xs, ys, linestyle='-', marker='.')
    plt.xlabel("Loss for legit")
    plt.ylabel("Accuracy")
    plt.show()


def main():
    # ns = [2, 3]
    alphas = [10 ** i for i in range(-5, 5, 2)]
    ns = [1, 2, 3]
    max_accuracy = 0.
    best_n = 1
    best_alpha = 1.
    for n in ns:
        for alpha in alphas:
            accuracy, _, _, _ = naive_bayes(n, alpha)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_n = n
                best_alpha = alpha
            print("n:", n, "\nalpha:", alpha, "\naccuracy:", accuracy)
            print('---------------------------------------')
    print('---------------------------------------')
    print("n:", best_n, "\nalpha:",  best_alpha, "\naccuracy:", max_accuracy)

    draw_roc(best_n, best_alpha)

    legit_loss_pow = 0
    for i in range(0, 200, 10):
        _, _, _, FP = naive_bayes(best_n, best_alpha, 10 ** i)
        if FP == 0:
            legit_loss_pow = i
            break
    print("legit loss: 10 ** ", legit_loss_pow)

    draw_accuracy_of_loss(best_n, best_alpha, legit_loss_pow)


parts_messages, parts_classes = read_all_parts()
max_n = 3
classes_counts_list = list()
ngrams_counts_list = list()
for train_messages, train_classes, test_messages, test_classes in k_fold(10):
    cl_cnts, ngrams_cnts = get_counts(max_n, train_messages, train_classes)
    classes_counts_list.append(cl_cnts)
    ngrams_counts_list.append(ngrams_cnts)
main()
