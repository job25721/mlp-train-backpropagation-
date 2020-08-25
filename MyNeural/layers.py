import numpy as np

class Node:
    def __init__(self,b, activation, name):
        self.input = None
        self.y = 0.0
        self.activation = activation
        self.w = []
        self.w_old = []
        self.b = b
        self.local_gradient = None
        self.name = name


    def updateWeight(self, weight,i):
        self.w[i] = weight

    def updateBias(self, bias):
        self.b = bias
    def showNodeDetail(self):
        print( "old(n-1) -------\n",self.w_old,"\ncurrent(n) ------\n",self.w, ", b :", self.b, ", input :",
              self.input,"y :",self.y,"local_grad :",self.local_gradient )

    def addInput(self, input):
        self.input = input

    def calculateOutput(self,prev_y):
        if len(prev_y ) != 0:
            if self.b == None:
                self.b = 0.0
            self.y =  self.activation(prev_y.dot(self.w) + self.b)
        else:
            if len(self.w) == 0:
                self.y = self.input
        return self.y



def Input(d):
    input_layer = []
    for i in range(d):
        node = Node(b=None, activation=None, name="Input")
        input_layer.append(node)
    return input_layer


def Dense(d, activation, name):
    hidden_layer = []
    for i in range(d):
        randBias = round(np.random.uniform(0.1, 1.0), 1)
        node = Node(b=randBias, activation=activation, name=name)
        hidden_layer.append(node)
    return hidden_layer


def Output(d, activation):
    output_layer = []
    for i in range(d):
        randBias = round(np.random.uniform(0.1, 1.0), 1)
        node = Node(randBias, activation, name="Output")
        output_layer.append(node)
    return output_layer


def layers_sumary(layers):
    for layer in layers:
        print("=================", layer[0].name, "==================")
        if type(layer) == list:
            i=1
            for n in layer:
                print(i,end=" : ")
                n.showNodeDetail()
                i += 1
        else:
            layer.showNodeDetail()
        print("=============================================")
