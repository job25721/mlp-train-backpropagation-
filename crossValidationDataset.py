import numpy as np
def crossVad():
    cross_pat = open('./dataset/cross.pat', 'r')
    crossValidation = []
    collect = []
    c = 0
    for data in cross_pat.readlines():
        collect.append(data.split("\n")[0])
        c += 1
        if c == 3:
            c = 0
            pattern = {
                "name": str(collect[0]),
                "feature": [float(collect[1].split(" ")[0]),float(collect[1].split(" ")[2])],
                "class": str(collect[2].split(" ")[0]) + str(collect[2].split(" ")[1])
            }
            crossValidation.append(pattern)
            collect = []
    return  crossValidation
