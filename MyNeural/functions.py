import math
import numpy as np
def normalization(dataset,max_x,min_x):
    normalized_data = []
    for x in dataset:
        normalized_z = []
        for xi in x:
            z = (xi - min_x) / (max_x - min_x)
            normalized_z.append(z)
        normalized_data.append(normalized_z)
    return normalized_data

def denormalize(z,max_x,min_x):
    return z * (max_x -min_x) + min_x

def prediction_error(desire_output, actual_output):
    err = ((desire_output - actual_output) ** 2)/2
    return err


def gradient_descent_output(old_weight, err,output):
    new_weight = old_weight - err*(output*(1-output))
    return new_weight
def gradient_descent_hidden(y):
    return y*(1-y)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def initilize_weight(node):
    randWeight = round(np.random.uniform(0.1, 1.0), 1)
    while node.w.__contains__(randWeight):
        randWeight = round(np.random.uniform(0.1, 1.0), 1)
    return randWeight