from MyNeural.functions import prediction_error, calc_new_weight, cross_validation_split, select_validate, calc_confusion_matrix, print_confusion_matrix, printProgressBar
from MyNeural.layers import layers_sumary
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import time


class Model:
    def __init__(self, input_layer, hidden_layers, output_layer):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

        self.layers = []

    def create_model(self):
        self.layers.append(self.input_layer)
        for layer in self.hidden_layers:
            self.layers.append(layer)
        self.layers.append(self.output_layer)
        # initialize weight
        for i, layer in enumerate(self.layers):
            if i != 0:
                for node in layer:
                    node.w = np.random.rand(len(self.layers[i-1]), 1)
                    node.w_old = np.zeros((len(self.layers[i-1]), 1))

    def sumary(self):
        layers = [self.input_layer]
        for h in self.hidden_layers:
            layers.append(h)
        layers.append(self.output_layer)
        layers_sumary(layers)

    def feed_forward(self, input_data):
        for l, layer in enumerate(self.layers):
            for i, node in enumerate(layer):
                if l == 0:
                    node.addInput(input_data[i])
                    node.calculateOutput([])
                else:
                    x = np.array([prev.y for prev in self.layers[l - 1]]
                                 ).reshape(1, len(self.layers[l - 1]))
                    node.calculateOutput(x)

    def back_propergation(self, err, momentum_rate, learning_rate):
        # find delta
        for l, layer in reversed(list(enumerate(self.layers))):
            if l != 0:
                for i, node in enumerate(layer):
                    if l == len(self.layers) - 1:
                        node.calculate_localgradient("output", err[i])
                    else:
                        delta_set = np.array(
                            [n.local_gradient for n in self.layers[l + 1]]).reshape((1, len(self.layers[l + 1])))
                        w_set = np.array([n.w[i] for n in self.layers[l + 1]])
                        node.calculate_localgradient(
                            "hidden", delta_set.dot(w_set))

        # update weight and bias
        for l, layer in reversed(list(enumerate(self.layers))):
            if l != 0:
                y_prev_set = []
                for prev_node in self.layers[l - 1]:
                    y_prev_set.append(prev_node.y)
                y_prev_set = np.array(y_prev_set).reshape(
                    len(self.layers[l - 1]), 1)
                for node in layer:
                    new_w = calc_new_weight(w=node.w, old_w=node.w_old, m_rate=momentum_rate,
                                            l_rate=learning_rate, y_prev=y_prev_set, local_grad=node.local_gradient)
                    new_b = node.b + \
                        (momentum_rate * (node.b - node.b_old)) + \
                        (1*learning_rate*node.local_gradient)
                    node.save_old_weight()
                    node.updateWeight(weight=new_w, bias=new_b)

    def Fit(self, dataset, epochs, momentum_rate, learning_rate, cross_validation, classification):
        cross_limit = 10
        train = []
        if cross_validation == 0:
            cross_limit = 1
            train = dataset
        else:
            cross_data = cross_validation_split(cross_validation, dataset)
            block = cross_data["data_block"]
            rand_set = cross_data["rand_set"]
            remiander_set = cross_data["rem_set"]
            print(rand_set)

        plot_data = []

        for c in range(cross_limit):
            if cross_validation != 0:
                res = select_validate(block, rand_set, c, remiander_set)
                train = res["train"]
                cross_valid = res["cross_valid"]
            printProgressBar(0, epochs, prefix='Training',
                             suffix='', length=25)
            for epoch in range(epochs):
                MSE = []
                # train
                for data in train:
                    input_data = np.array(data["input"])
                    desire_output = np.array(data["desire_output"])
                    # forward
                    self.feed_forward(input_data)
                    err = []
                    # find error
                    for i, node in enumerate(self.output_layer):
                        e = prediction_error(desire_output[i], node.y)
                        err.append(e)
                        MSE.append(np.power(e, 2))
                    # back propergate
                    self.back_propergation(err, momentum_rate, learning_rate)
                    # break
                MSE_result = np.average(MSE)
                printProgressBar(epoch + 1, epochs, prefix=f'Training epcoh:{epoch+1},cross:{c+1}(MSE: {MSE_result}, val_acc: {1 - MSE_result})',
                                 suffix='', length=25)
                np.random.shuffle(train)
            # cross validation
            if cross_validation != 0:
                MSE = []
                for cross_data in cross_valid:
                    input_data = np.array(cross_data["input"])
                    desire_output = np.array(cross_data["desire_output"])
                    # feed forward
                    self.feed_forward(input_data)
                    # find MSE
                    for i, node in enumerate(self.output_layer):
                        e = np.power(prediction_error(
                            desire_output[i], node.y), 2)
                        MSE.append(e)
                plot_data.append(np.average(MSE))
        if classification:
            confusion_matrix = {
                "01": {"t": 0, "f": 0},
                "10": {"t": 0, "f": 0}
            }
            for data in train:
                input_data = np.array(data["input"])
                desire_output = np.array(data["desire_output"])
                self.feed_forward(input_data)
                calc_confusion_matrix(
                    [node.y for node in self.output_layer], desire_output, confusion_matrix)
            print_confusion_matrix(confusion_matrix, len(train))
        print("alpha", momentum_rate, "etha", learning_rate)
        plt.title('Cross validation MSE')
        plt.plot(plot_data)
        plt.show()
