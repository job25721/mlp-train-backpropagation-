from dataset.floodDataset import readFloodDataset , cross
from MyNeural.functions import normalization, sigmoid
from MyNeural.model import Model
from MyNeural.layers import Input, Dense, Output
from random import shuffle
from time import sleep
import numpy as np

# # ============= read dataset===================
# f = open('./dataset/Flood_dataset.txt', 'r')
# floodDataset = readFloodDataset(file=f)
#
#
# # ==============================================
# # normalization
# train_dataset = [data["station1"] + data["station2"] + [data["desireOutput"]] for data in floodDataset]
# max_x = np.array(train_dataset).max()
# min_x = np.array(train_dataset).min()
#
# train_dataset = normalization(train_dataset,max_x,min_x)
#
# x = []
# for data in train_dataset:
#     x.append({
#         "input" : data[0:8],
#         "desire_output" : [data[8]]
#     })
#
#
#

f = open('./dataset/cross.pat','r')
cross = cross(file=f)
# # create a neural networks
print(len(cross[0]["input"]))
InputLayer = Input(d=len(cross[0]["input"]))
h1 = Dense(d=9, activation=sigmoid, name='h1')
#
OutputLayer = Output(d=1, activation=sigmoid)
#
# create model
my_model = Model(input_layer=InputLayer, hidden_layers=[
    h1], output_layer=OutputLayer)
# # my_model.sumary()
m = round(np.random.uniform(0.1, 0.9), 1)
l = round(np.random.uniform(0.1, 0.9), 1)
# # train model
# my_model.Fit(dataset=x, epochs=1000,
#              momentum_rate=m, learning_rate=l,cross_validation=0.1)
# # my_model.sumary()
# print("alpha =", m, "etha =", l)

my_model.Fit(dataset=cross,epochs=1000,momentum_rate=m,learning_rate=l,cross_validation=0.1)