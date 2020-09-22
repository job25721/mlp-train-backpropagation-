import math
import numpy as np


def normalization(dataset, max_x, min_x):
    normalized_data = []
    for x in dataset:
        normalized_z = []
        for xi in x:
            z = (xi - min_x) / (max_x - min_x)
            normalized_z.append(z)
        normalized_data.append(normalized_z)
    return normalized_data


def denormalize(z, max_x, min_x):
    return z * (max_x - min_x) + min_x


def prediction_error(desire_output, actual_output):
    err = desire_output - actual_output
    return err


def calc_new_weight(w, old_w, m_rate, l_rate, y_prev, local_grad):
    return w + (m_rate * (w-old_w)) + (y_prev.dot(l_rate*local_grad))


def local_gradient_output(y, err):
    return err * (y * (1-y))


def local_gradient_hidden(y, summation_next_local_gradient):
    return y * (1 - y) * summation_next_local_gradient


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def classificationRound(o1, o2):
    if o1 > o2:
        return [1, 0]
    else:
        return [0, 1]


def calc_confusion_matrix(output, desire_output, confusion_matrix):
    predicted = classificationRound(output[0], output[1])
    a = predicted[0]
    b = predicted[1]
    if(predicted == list(desire_output)):
        confusion_matrix[str(a)+str(b)]["t"] += 1
    else:
        confusion_matrix[str(a)+str(b)]["f"] += 1


def print_confusion_matrix(confusion_matrix, n):
    print("=== confusion matrix ===")
    print("\t0,1\t1,0")
    print(
        f'0,1\t{confusion_matrix["01"]["t"]}\t{confusion_matrix["01"]["f"]}\n1,0\t{confusion_matrix["10"]["f"]}\t{confusion_matrix["10"]["t"]}')
    r = (confusion_matrix["01"]["t"] +
         confusion_matrix["10"]["t"]) / n
    print(f"accuracy = {r}")


def cross_validation_split(cross_validate_num, dataset):
    cross_len = int(round((len(dataset) * cross_validate_num), 0))
    block = []
    random_set = []
    remainder_set = []
    for i in range(10):
        rand = np.random.randint(0, 9)
        while random_set.__contains__(rand):
            rand = np.random.randint(0, 10)
        random_set.append(rand)
        block.append(dataset[i * cross_len:cross_len + (i * cross_len)])
        if i == 9 and sum([len(b) for b in block]) < len(dataset):
            remainder_set = dataset[cross_len +
                                    (i * cross_len):len(dataset)]
    return {
        "data_block": block,
        "rand_set": random_set,
        "rem_set": remainder_set
    }


def select_validate(block, random_set, c, rem_set):
    cross_valid = block[random_set[c]]
    train_idx = random_set.copy()
    train_idx.remove(random_set[c])
    train = []
    for n in range(9):
        train += block[train_idx[n]]
        if c == 9 and n == 8:
            train += rem_set
    return {
        "train": train,
        "cross_valid": cross_valid
    }


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
