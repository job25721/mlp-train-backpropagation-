import numpy as np
from MyNeural.functions import local_gradient_output, local_gradient_hidden


class Node:
    def __init__(self, activation, name):
        self.input = None
        self.y = 0.0
        self.activation = activation
        self.w = []
        self.w_old = []
        self.b_old = 0
        self.b = 0
        self.local_gradient = None
        self.name = name

    def save_old_weight(self):
        self.w_old = self.w
        self.b_old = self.b

    def updateWeight(self, weight, bias):
        self.w = weight
        self.b = bias

    def showNodeDetail(self):
        print("old(n-1) -------\n", self.w_old, "\ncurrent(n) ------\n", self.w, ", b :", self.b, ", input :",
              self.input, "y :", self.y, "local_grad :", self.local_gradient)

    def addInput(self, input):
        self.input = input

    def calculate_localgradient(self, type, err):
        if type == "output":
            self.local_gradient = local_gradient_output(self.y, err)
        elif type == "hidden":
            self.local_gradient = local_gradient_hidden(self.y, err)[0][0]

    def calculateOutput(self, prev_y):
        if len(prev_y) != 0:
            self.y = self.activation(prev_y.dot(self.w) + self.b)
        else:
            if len(self.w) == 0:
                self.y = self.input
        return self.y


def Input(d):
    input_layer = []
    for i in range(d):
        node = Node(activation=None, name="Input")
        input_layer.append(node)
    return input_layer


def Dense(d, activation, name):
    hidden_layer = []
    for i in range(d):
        node = Node(activation=activation, name=name)
        hidden_layer.append(node)
    return hidden_layer


def Output(d, activation):
    output_layer = []
    for i in range(d):
        node = Node(activation=activation, name="Output")
        output_layer.append(node)
    return output_layer


def layers_sumary(layers):
    for layer in layers:
        print("=================", layer[0].name, "==================")
        if type(layer) == list:
            i = 1
            for n in layer:
                print(i, end=" : ")
                n.showNodeDetail()
                i += 1
        else:
            layer.showNodeDetail()
        print("=============================================")
