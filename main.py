from dataset.floodDataset import readFloodDataset
from MyNeural.functions import normalization, sigmoid
from MyNeural.model import Model
from MyNeural.layers import Input, Dense, Output
from random import shuffle
from time import sleep
import numpy as np

# ============= read dataset===================
f = open('./dataset/Flood_dataset.txt', 'r')
floodDataset = readFloodDataset(file=f)


# ==============================================
# normalization
train_dataset = [data["station1"] + data["station2"] + [data["desireOutput"]] for data in floodDataset]
max_x = np.array(train_dataset).max()
min_x = np.array(train_dataset).min()

train_dataset = normalization(train_dataset,max_x,min_x)



# create a neural networks
InputLayer = Input(d=8)
h1 = Dense(d=3, activation=sigmoid, name='h1')
h2 = Dense(d=3,activation=sigmoid, name='h2')
OutputLayer = Output(d=1, activation=sigmoid)

# create model
my_model = Model(input_layer=InputLayer, hidden_layers=[
    h1,h2], output_layer=OutputLayer, dataset_min=min_x, dataset_max=max_x)
# my_model.sumary()

# x = np.random.rand(2,1)
# print(x)
# x = np.zeros((1,2))
# y =np.zeros((2,1))
# print(x.dot(y))
m = round(np.random.uniform(0.1, 0.9), 1)
l = round(np.random.uniform(0.1, 0.9), 1)
# train model
my_model.Fit(dataset=train_dataset, epochs=1,
             momentum_rate=m, learning_rate=l,cross_validation=0.1)
# my_model.sumary()
# print("alpha =", m, "etha =", l)
# print("train complete...")
# sleep(3)
# my_model.sumary()
#
#
# group = [data["station1"] + data["station2"] +
#          [data["desireOutput"]] for data in testDataset]
#
# for data in group:
#     data[0] = 0
#     for i,d in enumerate(data):
#         if d > my_model.dataset_max:
#             print(data[i])
#             data[i] = my_model.dataset_max
#
# for i, data in enumerate(normalization(group, max_x, min_x), start=0):
#     testDataset[i]["station1"] = list(data[0:4])
#     testDataset[i]["station2"] = list(data[4:8])
#     testDataset[i]["desireOutput"] = data[8]
# my_model.forward(trainDataset)
