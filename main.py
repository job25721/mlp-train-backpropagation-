from dataset.floodDataset import readFloodDataset
from MyNeural.functions import normalization, sigmoid
from MyNeural.model import Model
from MyNeural.layers import Input, Dense, Output
from random import shuffle
from time import sleep
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
h1 = Dense(d=3, activation=sigmoid, name='h1')
h2 = Dense(d=2, activation=sigmoid, name='h2')
h3 = Dense(d=2, activation=sigmoid, name='h3')
OutputLayer = Output(d=1, activation=sigmoid)

# create model
my_model = Model(input_layer=InputLayer, hidden_layers=[
                 h1,h2,h3], output_layer=OutputLayer, dataset_min=min_x, dataset_max=max_x)
# my_model.sumary()

# train model
my_model.Fit(train_dataset=trainDataset, epochs=200)
my_model.sumary()