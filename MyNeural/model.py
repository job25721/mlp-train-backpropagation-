from MyNeural.functions import prediction_error ,denormalize
from MyNeural.layers import layers_sumary
from random import shuffle
import os
from time import sleep

class Model:
    def __init__(self, input_layer, hidden_layers, output_layer,dataset_min,dataset_max):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.dataset_max = dataset_max
        self.dataset_min = dataset_min

    def sumary(self):
        layers = [self.input_layer]
        for h in self.hidden_layers:
            layers.append(h)
        layers.append(self.output_layer)
        layers_sumary(layers)

    def Fit(self, train_dataset, epochs):
        for i in range(epochs):
            #find grad here -> update w here
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
                    for hidden_node in hidden_layer:
                        if count == 0:
                            hidden_node.addInput(sum([node.output for node in self.input_layer]))
                            hidden_node.calculateOutput()
                        else:
                            for node in self.hidden_layers[count - 1]:
                                hidden_node.addInput(sum([node.output for node in self.hidden_layers[count - 1]]))
                                hidden_node.calculateOutput()
                # outputLayers
                for output_node in self.output_layer:
                    output_node.addInput(sum([node.output for node in self.hidden_layers[len(self.hidden_layers) - 1]]))
                    output_node.calculateOutput()
                x = ""
                for n in range(50):
                    x += "="
                    # os.system('cls')
                    # print("TRAINING EPOCH", i + 1, "...")
                    print("[", x + "=>", "]",end='\r')
                    sleep(0.00001)
                x = ""
                for n in range(53):
                    x += "="
                print("[", x, "]")
                err = prediction_error(data["desireOutput"], output_node.output)
                print("desire output : ", data["desireOutput"], "actual output : ", output_node.output)
                print("desire output : ", denormalize(data["desireOutput"],self.dataset_max,self.dataset_min) , "actual output : ", denormalize(output_node.output,self.dataset_max,self.dataset_min))
                print("err : ", err * 100, "%")
                break

            shuffle(train_dataset)
