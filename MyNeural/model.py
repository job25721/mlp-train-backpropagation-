from MyNeural.functions import prediction_error, denormalize, local_gradient_output, local_gradient_hidden
from MyNeural.layers import layers_sumary
from random import shuffle
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
class Model:
    def __init__(self, input_layer, hidden_layers, output_layer):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

        self.layers = []
        self.create_model()

    def create_model(self):
        self.layers.append(self.input_layer)
        for layer in self.hidden_layers:
            self.layers.append(layer)
        self.layers.append(self.output_layer)

        # initialize weight
        for i,layer in enumerate(self.layers):
            if i != 0:
                for node in layer:
                    node.w = np.random.rand(len(self.layers[i-1]),1)
                    node.w_old = np.zeros((len(self.layers[i-1]),1))

    def sumary(self):
        layers = [self.input_layer]
        for h in self.hidden_layers:
            layers.append(h)
        layers.append(self.output_layer)
        layers_sumary(layers)

    def Fit(self, dataset, epochs, momentum_rate, learning_rate,cross_validation):

        cross_len = int(round((len(dataset)*0.1),0))
        # print(dataset)
        block = []
        random_set = []
        few_dataset = []
        for i in range(10):
            rand = np.random.randint(0, 9)
            while random_set.__contains__(rand):
                rand = np.random.randint(0, 10)
            random_set.append(rand)
            block.append(dataset[i*cross_len:cross_len+(i*cross_len)])
            if i == 9 and sum([len(b) for b in block]) < len(dataset) :
                few_dataset = dataset[cross_len+(i*cross_len):len(dataset)]

        plot_data = []
        for c in range(10):
            cross_valid = block[random_set[c]]
            train = []
            train_idx_set = []
            for n in range(9):
                rand = np.random.randint(0, 9)
                while rand == random_set[c] or train_idx_set.__contains__(rand):
                    rand = np.random.randint(0, 10)
                train_idx_set.append(rand)
                train += block[rand]
                if c == 9 and n == 8:
                    train += few_dataset
            for epoch in range(epochs):
                sum_sq_err = []
                print("EPOCH(c",c+1,"):",epoch+1,"====================================================")
                confusion = []
                all = []
                for z,data in enumerate(train):
                    input_data = np.array(data["input"])
                    desire_output = np.array(data["desire_output"])
                    # forward
                    for l, layer in enumerate(self.layers):
                        for i, node in enumerate(layer):
                            if l == 0:
                                node.addInput(input_data[i])
                                node.calculateOutput([])
                            else:
                                x = np.array([prev.y for prev in self.layers[l - 1]]).reshape(1, len(self.layers[l - 1]))
                                node.calculateOutput(x)

                    # print(z,self.layers[len(self.layers)-1][0].y)
                    err = []

                    for i, node in enumerate(self.output_layer):
                        # print(desire_output[i],node.y)
                        all.append(1)
                        if desire_output[i] == node.y:
                            confusion.append(1)

                        err.append(prediction_error(desire_output[i],node.y))
                    sum_sq_err.append(np.sum(err) ** 2)

                    # back propergate
                    # find delta
                    for l, layer in reversed(list(enumerate(self.layers))):
                        if l != 0:
                            for i, node in enumerate(layer):
                                if l == len(self.layers) - 1:
                                    node.local_gradient = local_gradient_output(node.y, err[i])
                                else:
                                    delta_set = np.array([n.local_gradient for n in self.layers[l + 1]]).reshape(
                                        (1, len(self.layers[l + 1])))
                                    w_set = np.array([n.w[i] for n in self.layers[l + 1]])
                                    node.local_gradient = local_gradient_hidden(node.y, delta_set.dot(w_set))[0][0]

                    # update weight
                    for l, layer in reversed(list(enumerate(self.layers))):
                        if l != 0:
                            y_prev_set = []
                            for prev_node in self.layers[l - 1]:
                                y_prev_set.append(prev_node.y)

                            y_prev_set = np.array(y_prev_set).reshape(len(self.layers[l - 1]), 1)
                            for node in layer:
                                new_w = node.w + (momentum_rate * (node.w - node.w_old)) +  (y_prev_set.dot(learning_rate * node.local_gradient))
                                node.w_old = node.w
                                node.w = new_w
                print("avg err :", np.average(sum_sq_err) * 100, "%")
                np.random.shuffle(train)
            sum_sq_err = []
            print("cross validate",c+1,"===================================")
            for cross_data in cross_valid:
                input_data = np.array(cross_data["input"])
                desire_output = np.array(cross_data["desire_output"])
                for l, layer in enumerate(self.layers):
                    for i, node in enumerate(layer):
                        if l == 0:
                            node.addInput(input_data[i])
                            node.calculateOutput([])
                        else:
                            x = np.array([prev.y for prev in self.layers[l - 1]]).reshape(1,len(self.layers[l - 1]))
                            node.calculateOutput(x)

                err = []
                for i, node in enumerate(self.output_layer):
                    err.append(prediction_error(desire_output[i], node.y))
                sum_sq_err.append(np.average(err) ** 2)
            plot_data.append(1 - np.average(sum_sq_err))
            print("avg cross validate err :", np.average(sum_sq_err) * 100, "%")

        print("alpha",momentum_rate,"etha",learning_rate)
        plt.title('Cross validation Accuracy')
        plt.plot(plot_data)
        plt.show()







