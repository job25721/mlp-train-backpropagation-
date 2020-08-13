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
