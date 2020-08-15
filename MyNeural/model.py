from MyNeural.functions import prediction_error ,denormalize , gradient_descent_output , initilize_weight
from MyNeural.layers import layers_sumary
from random import shuffle
import numpy as np
import os
from time import sleep

class Model:
    def __init__(self, input_layer, hidden_layers, output_layer,dataset_min,dataset_max):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.dataset_max = dataset_max
        self.dataset_min = dataset_min
        self.create_model()

    def create_model(self):
        for input_node in self.input_layer:
            for i in range(len(self.hidden_layers[0])):
                input_node.w.append(initilize_weight(node=input_node))

        for l,hidden_layer in enumerate(self.hidden_layers):
            for hidden_node in hidden_layer:
                if l == len(self.hidden_layers) - 1:
                    for i in self.output_layer:
                        hidden_node.w.append(initilize_weight(node=hidden_node))
                else:
                    for i in range(len(self.hidden_layers[l + 1])):
                        hidden_node.w.append(initilize_weight(node=hidden_node))

    def sumary(self):
        layers = [self.input_layer]
        for h in self.hidden_layers:
            layers.append(h)
        layers.append(self.output_layer)
        layers_sumary(layers)

    def Fit(self, train_dataset, epochs):
        for i in range(epochs):

            print("-----------------------------------------------------------")
            print("TRAINING EPOCH", i + 1, "...")
            for data in train_dataset:
                d = data["station1"] + data["station2"]
                # inputLayers
                for idx, input_node in enumerate(self.input_layer, start=0):
                    input_node.addInput(d[idx])
                    input_node.calculateOutput()

                # hiddenLayers
                for count, hidden_layer in enumerate(self.hidden_layers, start=0):
                    for i,hidden_node in enumerate(hidden_layer,start=0):
                        if count == 0:
                            hidden_node.addInput(sum([node.output[i] for node in self.input_layer]))
                            hidden_node.calculateOutput()
                        else:
                            hidden_node.addInput(sum([node.output[i] for node in self.hidden_layers[count - 1]]))
                            hidden_node.calculateOutput()

                # outputLayers
                for i,output_node in enumerate(self.output_layer,start=0):
                    output_node.addInput(sum([node.output[i] for node in self.hidden_layers[len(self.hidden_layers) - 1]]))
                    output_node.calculateOutput()
                loading_idicator = ""
                for n in range(50):
                    loading_idicator += "="
                    # os.system('cls')
                    # print("TRAINING EPOCH", i + 1, "...")
                    print("[", loading_idicator + "=>", "]",end='\r')
                    sleep(0.00001)
                err = prediction_error(data["desireOutput"], output_node.output)
                # print("desire output : ", data["desireOutput"], "actual output : ", output_node.output)
                print("desire output : ", denormalize(data["desireOutput"],self.dataset_max,self.dataset_min) , "actual output : ", denormalize(output_node.output,self.dataset_max,self.dataset_min))
                print("err : ", err * 100, "%")
                # backpropergation
                j = len(self.hidden_layers) - 1
                for i, output_node in enumerate(self.output_layer, start=0):
                    for hidden_node in self.hidden_layers[j]:
                        old_weight = hidden_node.w[i]
                        output = output_node.output
                        new_weight = gradient_descent_output(old_weight, err, output)
                        hidden_node.updateWeight(weight=new_weight, i=i)
                while j >= 0:
                    for i,hidden_node in enumerate(self.hidden_layers[j],start=0):
                        if j-1 < 0:
                            for node in self.input_layer:
                                output = hidden_node.y
                                old_weight = node.w[i]
                                #find grad this line
                                new_weight = 0.0
                                node.updateWeight(new_weight,i)
                        else:
                            for prev_node in self.hidden_layers[j-1]:
                                old_weight = prev_node.w[i]
                                output = hidden_node.y
                                new_weight = 0.0
                                prev_node.updateWeight(new_weight,i)
                        j -= 1
                break # read one line dataset for test

            shuffle(train_dataset)
