def readFloodDataset(file):
    file.readline()
    file.readline()
    dataSet = []
    c = 0
    for f in file.readlines():
        line = f.split("\t")
        s1 = [float(data) for data in line[0:4]]
        s2 = [float(data) for data in line[4:8]]
        s1.reverse()
        s2.reverse()
        dataSet.append({
            "id": c,
            "station1": s1,
            "station2": s2,
            "desireOutput": float(line[8].split('\n')[0])
        })
        c += 1
    return dataSet


def cross(file):
    crossValidation = []
    collect = []
    c = 0
    for data in file.readlines():
        collect.append(data.split("\n")[0])
        c += 1
        if c == 3:
            c = 0
            pattern = {
                "name": str(collect[0]),
                "input": [float(collect[1].split(" ")[0]), float(collect[1].split(" ")[2])],
                "desire_output": [int(collect[2].split(" ")[0]), int(collect[2].split(" ")[1])]
            }
            crossValidation.append(pattern)
            collect = []
    return crossValidation
