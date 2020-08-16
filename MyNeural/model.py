from MyNeural.functions import prediction_error, denormalize, local_gradient_output, local_gradient_hidden,  initialize_weight , update_weight
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
        #initialize weight
        for l, hidden_layer in enumerate(self.hidden_layers):
            for hidden_node in hidden_layer:
                if l == 0:
                    for i in range(len(self.input_layer)):
                        hidden_node.w.append(initialize_weight(node=hidden_node))
                        hidden_node.w_old.append(0.0)
                else:
                    for i in range(len(self.hidden_layers[l - 1])):
                        hidden_node.w.append(initialize_weight(node=hidden_node))
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

    def Fit(self, train_dataset, epochs,momentum_rate,learning_rate):
        loading_idicator = ""
        # print("[", loading_idicator + "=>", "]", end='\r')
        for epoch in range(epochs):
            print("--------------------------------------------------------------------")
            print("EPOCH", epoch + 1, "...")
            if epoch % 10 == 0:
                loading_idicator += "="
            print("[Loading ", loading_idicator + "=>", "]", end='\r')
            avg_err = 0.0
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
                            hidden_node.addInput(sum([node.y for node in self.input_layer]))
                            hidden_node.calculateOutput()
                        else:
                            hidden_node.addInput(sum([node.y for node in self.hidden_layers[count - 1]]))
                            hidden_node.calculateOutput()

                # outputLayer
                for i, output_node in enumerate(self.output_layer, start=0):
                    output_node.addInput(sum([node.y for node in self.hidden_layers[len(self.hidden_layers) - 1]]))
                    output_node.calculateOutput()
                err = prediction_error(data["desireOutput"], output_node.y)
                # print("desire output : ",
                #       data["desireOutput"], "actual output : ", output_node.y)
                # print("desire output : ", denormalize(data["desireOutput"], self.dataset_max, self.dataset_min),
                #       "actual output : ", denormalize(
                #         output_node.y, self.dataset_max, self.dataset_min))
                # print("err : ", err * 100, "%")
                avg_err += err
                # backpropergation

                # find local gradient
                for output_node in self.output_layer:
                    output_node.local_gradient = local_gradient_output(err=err, y=output_node.y)

                for i, hidden_layer in reversed(list(enumerate(self.hidden_layers))):
                    for j,hidden_node in enumerate(hidden_layer,start=0):
                        summation = 0.0
                        if i == len(self.hidden_layers) - 1:
                            for output_node in self.output_layer:
                                summation += output_node.local_gradient * output_node.w[j]
                        else:
                            for node in self.hidden_layers[i+1]:
                                summation += node.local_gradient * node.w[j]
                        hidden_node.local_gradient = local_gradient_hidden(hidden_node.y,summation)

                for i, input_node in enumerate(self.input_layer,start=0):
                    summation = 0.0
                    for node in self.hidden_layers[0]:
                        summation += node.local_gradient * node.w[i]
                    input_node.local_gradient = local_gradient_hidden(input_node.y,summation)

                #update weight
                for i,hidden_layer in reversed(list(enumerate(self.hidden_layers))):
                    for k,hidden_node in enumerate(hidden_layer,start=0): #select each y-1
                        prev_y = hidden_node.y
                        if i == len(self.hidden_layers) - 1:
                            for output_node in self.output_layer:
                                new_w = update_weight(current_w=output_node.w[k],old_w=output_node.w_old[k],
                                                     local_gradient=output_node.local_gradient,y=prev_y,
                                                     alpha=momentum_rate,etha=learning_rate)
                                output_node.w_old[k] = output_node.w[k]
                                output_node.updateWeight(new_w,k)
                        else:
                            for node in self.hidden_layers[i+1]:
                                new_w = update_weight(current_w=node.w[k],old_w=node.w_old[k],
                                                      local_gradient=node.local_gradient,y=prev_y,
                                                      alpha=momentum_rate,etha=learning_rate)
                                node.w_old[k] = node.w[k]
                                node.updateWeight(new_w,k)

                for k,input_node in enumerate(self.input_layer,start=0): #select each y-1
                    prev_y = input_node.y
                    for node in self.hidden_layers[0]:
                        new_w = update_weight(current_w=node.w[k],old_w=node.w_old[k],
                                              local_gradient=node.local_gradient,y=prev_y,
                                              alpha=momentum_rate,etha=learning_rate)
                        node.w_old[k] = node.w[k]
                        node.updateWeight(new_w,k)
                # break  # read one line dataset for test
            avg_err = avg_err / len(train_dataset)
            print("avg err :",avg_err *100 ,"%")
            shuffle(train_dataset)


