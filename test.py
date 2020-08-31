import numpy as np

x = [[1,2]]
y = [[3,4]]

x = np.array(x)
y = np.array(y).reshape((2,1))
print(x.dot(y))

# print(y)