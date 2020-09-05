from dataset.floodDataset import readFloodDataset, cross
from MyNeural.functions import normalization, sigmoid
from MyNeural.model import Model
from MyNeural.layers import Input, Dense, Output
from random import shuffle
from time import sleep
import numpy as np

def random_rate():
    m = round(np.random.uniform(0.1, 0.9), 1)
    l = round(np.random.uniform(0.1, 0.9), 1)
    return {"m_rate" : m , "l_rate" : l}

def floodModel():
    # ============= read dataset===================
    f = open('./dataset/Flood_dataset.txt', 'r')
    floodDataset = readFloodDataset(file=f)
    # ==============================================
    # normalization
    train_dataset = [data["station1"] + data["station2"] + [data["desireOutput"]] for data in floodDataset]
    max_x = np.array(train_dataset).max()
    min_x = np.array(train_dataset).min()
    train_dataset = normalization(train_dataset, max_x, min_x)
    x = []
    for data in train_dataset:
        x.append({
            "input": data[0:8],
            "desire_output": [data[8]]
        })
    np.random.shuffle(x)
    # create a neural networks
    InputLayer = Input(d=8)
    h1 = Dense(d=3, activation=sigmoid, name='h1')
    OutputLayer = Output(d=1, activation=sigmoid)

    # create model
    my_model = Model(input_layer=InputLayer, hidden_layers=[h1], output_layer=OutputLayer)
    my_model.sumary()

    # train model
    my_model.Fit(dataset=x, epochs=1000,momentum_rate=random_rate()["m_rate"], learning_rate=random_rate()["l_rate"],cross_validation=0.1)
    my_model.sumary()



def crossTest():
    f = open('./dataset/cross.pat', 'r')
    dataset = cross(file=f)
    InputLayer = Input(d=2)
    h1 = Dense(d=3,activation=sigmoid,name='h1')
    h2 = Dense(d=2,activation=sigmoid,name='h2')
    OutputLayer = Output(d=2,activation=sigmoid)

    cross_model = Model(input_layer=InputLayer,hidden_layers=[h1,h2],output_layer=OutputLayer)
    cross_model.Fit(dataset=dataset,epochs=1000,momentum_rate=random_rate()["m_rate"],learning_rate=random_rate()["l_rate"],cross_validation=0.1)



crossTest()

