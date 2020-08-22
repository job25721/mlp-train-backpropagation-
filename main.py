from dataset.floodDataset import readFloodDataset
from MyNeural.functions import normalization, sigmoid
from MyNeural.model import Model
from MyNeural.layers import Input, Dense, Output
from random import shuffle
from time import sleep
from numpy import random

# ============= read dataset===================
f = open('./dataset/Flood_dataset.txt', 'r')
floodDataset = readFloodDataset(file=f)
# shuffle(floodDataset)
train_length = int(round(len(floodDataset) * 0.9, 0))
trainDataset = floodDataset[0:train_length]
testDataset = floodDataset[train_length:len(floodDataset)]
# ==============================================
# normalization
group = [data["station1"] + data["station2"] +
         [data["desireOutput"]] for data in trainDataset]
tmp = []
for s in group:
    for d in s:
        tmp.append(d)
max_x = max(tmp)
min_x = min(tmp)

for i, data in enumerate(normalization(group, max_x, min_x), start=0):
    trainDataset[i]["station1"] = list(data[0:4])
    trainDataset[i]["station2"] = list(data[4:8])
    trainDataset[i]["desireOutput"] = data[8]
# print(trainDataset)

# create a neural networks
InputLayer = Input(d=8)
h1 = Dense(d=120, activation=sigmoid, name='h1')
OutputLayer = Output(d=1, activation=sigmoid)

# create model
my_model = Model(input_layer=InputLayer, hidden_layers=[
    h1], output_layer=OutputLayer, dataset_min=min_x, dataset_max=max_x)
my_model.sumary()

m = round(random.uniform(0.1, 0.9), 1)
l = round(random.uniform(0.1, 0.9), 1)
# train model
my_model.Fit(train_dataset=trainDataset, epochs=1,
             momentum_rate=1, learning_rate=l)
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
