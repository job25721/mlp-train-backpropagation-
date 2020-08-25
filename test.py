import numpy as np

random_set = []
x = [0,1,2,3,4,5,6,7,8,9]
lenc = 1
for c in range(len(x)):
    rand = np.random.randint(0, 9)
    while random_set.__contains__(rand):
        rand = np.random.randint(0, 10)
    random_set.append(rand)
    print(x[random_set[c]*lenc:lenc+(random_set[c]*lenc)])

print(random_set)