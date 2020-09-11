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
