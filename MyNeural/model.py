from MyNeural.functions import prediction_error, denormalize, local_gradient_output, local_gradient_hidden
from MyNeural.layers import layers_sumary
from random import shuffle
import numpy as np


class Model:
    def __init__(self, input_layer, hidden_layers, output_layer, dataset_min, dataset_max):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.dataset_max = dataset_max
        self.dataset_min = dataset_min
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



        for c in range(10):
            cross_valid = block[random_set[c]]
            train = []
            for n in range(9):
                rand = np.random.randint(0, 9)
                while rand == c:
                    rand = np.random.randint(0, 10)
                train.append(block[rand])



            # for epoch in range(epochs):
            #
            #     print("EPOCH:",epoch+1,"====================================================")
            #     for z,data in enumerate(train):
            #         input_data = data[0:8]
            #         desire_output = [data[8]]
            #
            #         # forward
            #         for l, layer in enumerate(self.layers):
            #             for i, node in enumerate(layer):
            #                 if l == 0:
            #                     node.addInput(input_data[i])
            #                     node.calculateOutput([])
            #                 else:
            #                     x = np.array([prev.y for prev in self.layers[l - 1]]).reshape(1, len(self.layers[l - 1]))
            #                     node.calculateOutput(x)
            #
            #         # print(z,self.layers[len(self.layers)-1][0].y)
            #         err = []
            #         for i, node in enumerate(self.output_layer):
            #             err.append(prediction_error(desire_output[i],node.y))
            #
            #
            #
            #
            #
            #         # back propergate
            #         # find delta
            #         for l, layer in reversed(list(enumerate(self.layers))):
            #             if l != 0:
            #                 for i, node in enumerate(layer):
            #                     if l == len(self.layers) - 1:
            #                         node.local_gradient = local_gradient_output(node.y, err[i])
            #                     else:
            #                         delta_set = np.array([n.local_gradient for n in self.layers[l + 1]]).reshape(
            #                             (1, len(self.layers[l + 1])))
            #                         w_set = np.array([n.w[i] for n in self.layers[l + 1]])
            #                         node.local_gradient = local_gradient_hidden(node.y, delta_set.dot(w_set))[0][0]
            #
            #         # update weight
            #         for l, layer in reversed(list(enumerate(self.layers))):
            #             if l != 0:
            #                 y_prev_set = []
            #                 for prev_node in self.layers[l - 1]:
            #                     y_prev_set.append(prev_node.y)
            #                 y_prev_set = np.array(y_prev_set).reshape(len(self.layers[l - 1]), 1)
            #                 for node in layer:
            #                     new_w = node.w + (momentum_rate * (node.w - node.w_old)) +  (y_prev_set.dot(learning_rate * node.local_gradient))
            #                     node.w_old = node.w
            #                     node.w = new_w
            #     sum_sq_err = []
            #
            #
            #     for cross_data in cross_vad:
            #         input_data = cross_data[0:8]
            #         desire_output = [cross_data[8]]
            #         for l, layer in enumerate(self.layers):
            #             for i, node in enumerate(layer):
            #                 if l == 0:
            #                     node.addInput(input_data[i])
            #                     node.calculateOutput([])
            #                 else:
            #                     x = np.array([prev.y for prev in self.layers[l - 1]]).reshape(1, len(self.layers[l - 1]))
            #                     node.calculateOutput(x)
            #
            #         err = []
            #         for i, node in enumerate(self.output_layer):
            #             err.append(prediction_error(desire_output[i], node.y))
            #
            #         sum_sq_err.append(np.average(err)**2)
            #
            #     print("avg err :", np.average(sum_sq_err) * 100, "%")






