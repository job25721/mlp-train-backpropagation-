from MyNeural.functions import prediction_error, denormalize, local_gradient_output, local_gradient_hidden, \
    initialize_weight
from MyNeural.layers import layers_sumary
from random import shuffle
import numpy as np
import os
from time import sleep


class Model:
    def __init__(self, input_layer, hidden_layers, output_layer, dataset_min, dataset_max):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.dataset_max = dataset_max
        self.dataset_min = dataset_min
        self.create_model()

    def create_model(self):
        for input_node in self.input_layer:
            for i in range(len(self.hidden_layers[0])):
                input_node.w.append(initialize_weight(node=input_node))
                input_node.output.append(0.0)

        for l, hidden_layer in enumerate(self.hidden_layers):
            for hidden_node in hidden_layer:
                if l == len(self.hidden_layers) - 1:
                    for i in self.output_layer:
                        hidden_node.w.append(initialize_weight(node=hidden_node))
                        hidden_node.output.append(0.0)
                else:
                    for i in range(len(self.hidden_layers[l + 1])):
                        hidden_node.w.append(initialize_weight(node=hidden_node))
                        hidden_node.output.append(0.0)

    def sumary(self):
        layers = [self.input_layer]
        for h in self.hidden_layers:
            layers.append(h)
        layers.append(self.output_layer)
        layers_sumary(layers)

    def Fit(self, train_dataset, epochs):
        loading_idicator = ""
        # print("[", loading_idicator + "=>", "]", end='\r')
        for epoch in range(epochs):
            print("--------------------------------------------------------------------")
            print("EPOCH", epoch + 1, "...")
            if epoch % 10 == 0:
                loading_idicator += "="
            print("[Loading ", loading_idicator + "=>", "]", end='\r')
            for data in train_dataset:
                d = data["station1"] + data["station2"]
                # inputLayer
                for idx, input_node in enumerate(self.input_layer, start=0):
                    input_node.addInput(d[idx])
                    input_node.calculateOutput()

                # hiddenLayers
                for count, hidden_layer in enumerate(self.hidden_layers, start=0):
                    for i, hidden_node in enumerate(hidden_layer, start=0):
                        if count == 0:
                            hidden_node.addInput(sum([node.output[i] for node in self.input_layer]))
                            hidden_node.calculateOutput()
                        else:
                            hidden_node.addInput(sum([node.output[i] for node in self.hidden_layers[count - 1]]))
                            hidden_node.calculateOutput()

                # outputLayer
                for i, output_node in enumerate(self.output_layer, start=0):
                    output_node.addInput(sum([node.output[i] for node in self.hidden_layers[len(self.hidden_layers) - 1]]))
                    output_node.calculateOutput()
                err = prediction_error(data["desireOutput"], output_node.output)
                # print("desire output : ",
                #       data["desireOutput"], "actual output : ", output_node.output)
                # print("desire output : ", denormalize(data["desireOutput"], self.dataset_max, self.dataset_min),
                #       "actual output : ", denormalize(
                #         output_node.output, self.dataset_max, self.dataset_min))
                print("err : ", err * 100, "%")

                # backpropergation

                # find local gradient
                for output_node in self.output_layer:
                    output_node.local_gradient = local_gradient_output(err=err, y=output_node.y)

                for i, hidden_layer in reversed(list(enumerate(self.hidden_layers))):
                    for hidden_node in hidden_layer:
                        summation = 0.0
                        if i == len(self.hidden_layers) - 1:
                            for j, output_node in enumerate(self.output_layer, start=0):
                                summation += (output_node.local_gradient * hidden_node.w[j])
                            hidden_node.local_gradient = local_gradient_hidden(hidden_node.y, summation)
                        else:
                            for j, node in enumerate(self.hidden_layers[i + 1]):
                                summation += (node.local_gradient * hidden_node.w[j])
                            hidden_node.local_gradient = local_gradient_hidden(hidden_node.y, summation)

                #update weight
                for i, hidden_layer in reversed(list(enumerate(self.hidden_layers))):
                    for hidden_node in hidden_layer:
                        if i == len(self.hidden_layers) - 1:
                            for j, output_node in enumerate(self.output_layer, start=0):
                                new_weight = hidden_node.w[j] - output_node.local_gradient
                                hidden_node.updateWeight(new_weight, j)
                        else:
                            for j, node in enumerate(self.hidden_layers[i + 1]):
                                new_weight = hidden_node.w[j] - node.local_gradient
                                hidden_node.updateWeight(new_weight, j)

                for input_node in self.input_layer:
                    for i, h1_node in enumerate(self.hidden_layers[0]):
                        new_weight = input_node.w[i] - h1_node.local_gradient
                        input_node.updateWeight(new_weight, i)

                # break  # read one line dataset for test

            shuffle(train_dataset)
