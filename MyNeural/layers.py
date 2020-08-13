from numpy import random


class Node:
    def __init__(self, w, b, activation, name):
        self.input = None
        self.output = None
        self.activation = activation
        self.w = w
        self.b = b
        self.name = name

    def updateWeight(self, weight):
        self.w = weight

    def updateBias(self, bias):
        self.b = bias

    def showNodeDetail(self):
        print("[ weight :", self.w, ", bias :", self.b, ", input :",
              self.input, ", output :", self.output, "]")

    def addInput(self, input):
        self.input = input

    def calculateOutput(self):
        if self.input == None:
            print("please add input")
        else:
            if self.b == None:
                self.b = 0.0
            if (self.activation != None):
                if self.w == None:  # outputLayer
                    self.output = self.activation(self.input + self.b)
                else:
                    self.output = self.w * self.activation(self.input + self.b)
            else:
                self.output = self.w * self.input
        return self.output


def Input(d):
    input_layer = []
    for i in range(d):
        randWeight = 0.0
        while randWeight == 0.0:
            randWeight = round(random.uniform(-0.9, 1.0), 1)
        node = Node(w=randWeight, b=None, activation=None, name="Input")
        input_layer.append(node)
    return input_layer


def Dense(d, activation, name):
    hidden_layer = []
    for i in range(d):
        randWeight = 0.0
        randBias = round(random.uniform(0.1, 1.0), 1)
        while randWeight == 0.0:
            randWeight = round(random.uniform(-0.9, 1.0), 1)
        node = Node(randWeight, randBias, activation, name)
        hidden_layer.append(node)
    return hidden_layer


def Output(d, activation):
    output_layer = []
    for i in range(d):
        randBias = round(random.uniform(0.1, 1.0), 1)
        node = Node(None, randBias, activation, name="Output")
        output_layer.append(node)
    return output_layer


def layers_sumary(layers):
    for layer in layers:
        print("=================", layer[0].name, "==================")
        if type(layer) == list:
            i = 1
            for n in layer:
                print(i, end=": ")
                n.showNodeDetail()
                i += 1
        else:
            layer.showNodeDetail()
        print("=============================================")
