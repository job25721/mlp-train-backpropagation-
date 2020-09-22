from dataset.read_dataset import readFloodDataset, cross
from MyNeural.functions import normalization, sigmoid
from MyNeural.model import Model
from MyNeural.layers import Input, Dense, Output
from random import shuffle
from time import sleep
import numpy as np


def random_rate():
    m = round(np.random.uniform(0.1, 0.9), 1)
    l = round(np.random.uniform(0.1, 0.9), 1)
    return {"m_rate": m, "l_rate": l}


def create_hidden(hidden_struct):
    hidden_layers = []
    for i, h in enumerate(hidden_struct.split("-")):
        layer_name = 'h' + str(i+1)
        l = Dense(d=int(h), activation=sigmoid, name=layer_name)
        hidden_layers.append(l)
    return hidden_layers


def floodModel(ep, struct):
    # ============= read dataset===================
    f = open('./dataset/Flood_dataset.txt', 'r')
    floodDataset = readFloodDataset(file=f)
    # ==============================================
    # normalization
    train_dataset = [data["station1"] + data["station2"] +
                     [data["desireOutput"]] for data in floodDataset]
    max_x = np.array(train_dataset).max()
    min_x = np.array(train_dataset).min()
    train_dataset = normalization(train_dataset, max_x, min_x)
    x = []
    for data in train_dataset:
        x.append({
            "input": data[0:8],
            "desire_output": [data[8]]
        })
    # create a neural networks
    InputLayer = Input(d=8)
    OutputLayer = Output(d=1, activation=sigmoid)

    # create model
    my_model = Model(input_layer=InputLayer, hidden_layers=create_hidden(
        struct), output_layer=OutputLayer)
    my_model.create_model()
    my_model.sumary()

    # train model
    my_model.Fit(dataset=x, epochs=ep, momentum_rate=random_rate()[
        "m_rate"], learning_rate=random_rate()["l_rate"], cross_validation=0.1, classification=False)
    my_model.sumary()


def crossTest(ep, struct):
    f = open('./dataset/cross.pat', 'r')
    dataset = cross(file=f)
    InputLayer = Input(d=2)
    OutputLayer = Output(d=2, activation=sigmoid)

    cross_model = Model(input_layer=InputLayer, hidden_layers=create_hidden(
        struct), output_layer=OutputLayer)
    cross_model.create_model()
    cross_model.Fit(dataset=dataset, epochs=ep, momentum_rate=random_rate()[
        "m_rate"], learning_rate=random_rate()["l_rate"], cross_validation=0.1, classification=True)


cmd = 'init'
while cmd != "c":
    cmd = input("1:flood train\n2:cross train\nc:cancel\ncmd > ")
    if cmd == "1":
        ep = int(input("epochs ? : "))
        struct = input("input hidden layers ex.2-3-1 : ")
        print("flood train")
        sleep(2)
        floodModel(ep, struct)
    elif cmd == "2":
        ep = int(input("epochs ? : "))
        struct = input("input hidden layers ex.2-3-1 : ")
        print("cross train")
        sleep(2)
        crossTest(ep, struct)
    elif cmd == "c":
        print("end process")
        break
