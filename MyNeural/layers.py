from numpy import random


class Node:
    def __init__(self,b, activation, name):
        self.input = None
        self.output = []
        self.y = 0
        self.activation = activation
        self.w = []
        self.b = b
        self.name = name


    def updateWeight(self, weight,i):
        self.w[i] = weight

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
                if self.w == []:  # outputLayer
                    self.y = self.activation(self.input + self.b)
                    self.output = self.y
                else:
                    self.y = self.activation(self.input + self.b)
                    for w in self.w:
                        self.output.append(w * self.y)

            else:
                self.y = self.input
                for w in self.w:
                    self.output.append(w * self.y)
        return self.output


def Input(d):
    input_layer = []
    for i in range(d):
        node = Node(b=None, activation=None, name="Input")
        input_layer.append(node)
    return input_layer


def Dense(d, activation, name):
    hidden_layer = []
    for i in range(d):
        randBias = round(random.uniform(0.1, 1.0), 1)
        node = Node(b=randBias, activation=activation, name=name)
        hidden_layer.append(node)
    return hidden_layer


def Output(d, activation):
    output_layer = []
    for i in range(d):
        randBias = round(random.uniform(0.1, 1.0), 1)
        node = Node(randBias, activation, name="Output")
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
