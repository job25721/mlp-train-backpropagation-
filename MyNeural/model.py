from MyNeural.functions import prediction_error, denormalize, local_gradient_output, local_gradient_hidden,  initialize_weight, update_weight
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
        # initialize weight
        for l, hidden_layer in enumerate(self.hidden_layers):
            for hidden_node in hidden_layer:
                if l == 0:
                    for i in range(len(self.input_layer)):
                        hidden_node.w.append(
                            initialize_weight(node=hidden_node))
                        hidden_node.w_old.append(0.0)
                else:
                    for i in range(len(self.hidden_layers[l - 1])):
                        hidden_node.w.append(
                            initialize_weight(node=hidden_node))
                        hidden_node.w_old.append(0.0)
        for hidden_node in self.hidden_layers[len(self.hidden_layers) - 1]:
            for output_node in self.output_layer:
                output_node.w.append(initialize_weight(node=output_node))
                output_node.w_old.append(0.0)

    def sumary(self):
        layers = [self.input_layer]
        for h in self.hidden_layers:
            layers.append(h)
        layers.append(self.output_layer)
        layers_sumary(layers)

    def Fit(self, train_dataset, epochs, momentum_rate, learning_rate):
        # loading_idicator = ""
        for epoch in range(epochs):
            print("--------------------------------------------------------------------")
            print("EPOCH", epoch + 1, "...")

            avg_err = 0.0

            for z,data in enumerate(train_dataset,start=0):

                d = data["station1"] + data["station2"]
                for idx,input_node in enumerate(self.input_layer):
                    input_node.addInput(d[idx])
                    input_node.calculateOutput([])

                for count,hidden_layer in enumerate(self.hidden_layers,start=0):
                    for node in hidden_layer:
                        if count == 0:
                            node.calculateOutput([prev_y.y for prev_y in self.input_layer])
                        else:
                            node.calculateOutput([prev_y.y for prev_y in self.hidden_layers[count-1]])

                for output_node in self.output_layer:
                    output_node.calculateOutput([prev_y.y for prev_y in self.hidden_layers[len(self.hidden_layers)-1]])


                err = prediction_error(desire_output=data["desireOutput"],actual_output=self.output_layer[0].y)
                avg_err += err


                print("output :",z,"=", self.output_layer[0].y)
                # if self.output_layer[0].y == 1.0:
                #     break

                #back popergation
                #find local gradient

                for output_node in self.output_layer:
                    output_node.local_gradient = local_gradient_output(y=output_node.y,err=err)

                for count,hidden_layer in reversed(list(enumerate(self.hidden_layers))):
                    for i,node in enumerate(hidden_layer):
                        summation = 0.0
                        if count == len(self.hidden_layers) - 1:
                            summation = sum([out_n.w[i] * out_n.local_gradient for out_n in self.output_layer])
                        else:
                            summation = sum([prev_node.w[i] * prev_node.local_gradient for prev_node in self.hidden_layers[count+1]])
                        node.local_gradient = local_gradient_hidden(node.y, summation)

                #update weight
                for node in self.output_layer:
                    for i,w in enumerate(node.w):
                        new_w = update_weight(current_w=w,old_w=node.w_old[i],
                                              local_gradient=node.local_gradient,
                                              y_prev=self.hidden_layers[len(self.hidden_layers) - 1][i].y,alpha=momentum_rate,etha=learning_rate)
                        node.w_old[i] = w
                        node.updateWeight(new_w,i)

                for l,hidden_layer in reversed(list(enumerate(self.hidden_layers))):
                    for node in hidden_layer:
                        for i, w in enumerate(node.w):
                            y_prev = 0.0
                            if l==0:
                                y_prev = self.input_layer[i].y
                            else:
                                y_prev = self.hidden_layers[l-1][i].y
                            new_w = update_weight(current_w=w, old_w=node.w_old[i],
                                                  local_gradient=node.local_gradient,
                                                  y_prev=y_prev,
                                                  alpha=momentum_rate, etha=learning_rate)
                            node.w_old[i] = w
                            node.updateWeight(new_w,i)

                # self.sumary()
                # if z==70:
                #     break

                # break  # read one line dataset for test

            # avg_err = avg_err / len(train_dataset)
            # print("avg err :", avg_err * 100, "%")
            shuffle(train_dataset)

    def forward(self, input_data):
        avg_err = 0.0
        for data in input_data:
            d = data["station1"] + data["station2"]
            # inputLayer
            for idx, input_node in enumerate(self.input_layer, start=0):
                input_node.addInput(d[idx])
                input_node.calculateOutput()

            # hiddenLayers
            for count, hidden_layer in enumerate(self.hidden_layers, start=0):
                for i, hidden_node in enumerate(hidden_layer, start=0):
                    if count == 0:
                        hidden_node.addInput(
                            sum([node.y for node in self.input_layer]))
                        hidden_node.calculateOutput()
                    else:
                        hidden_node.addInput(
                            sum([node.y for node in self.hidden_layers[count - 1]]))
                        hidden_node.calculateOutput()

            # outputLayer
            for i, output_node in enumerate(self.output_layer, start=0):
                output_node.addInput(
                    sum([node.y for node in self.hidden_layers[len(self.hidden_layers) - 1]]))
                output_node.calculateOutput()
            err = prediction_error(
                data["desireOutput"], self.output_layer[0].y)
            print(err)
            avg_err += err
        print("avg err : ", avg_err/len(test_dataset))
