import numpy as np
import random
import math
from crop import crop

# file = crop("dep_out_data_zoom.npy")
file = "pd_out_data_zoom.npy"

# print(file)
fa_and_pd_new = np.load(file, allow_pickle=True)
np.random.shuffle(fa_and_pd_new)
x = [i[0] for i in fa_and_pd_new]
y = [i[1] for i in fa_and_pd_new]
# np.save("x.npy",np.array(x,dtype=object))
# np.save("y.npy",np.array(y,dtype=object))

dep_count = math.ceil(0.8*y.count(1))
nor_count = int(0.8*y.count(0))
# print(y)
# print(dep_count,nor_count)
# print(y.count(1),y.count(0))
x_train = []
y_train = []
x_test = []
y_test = []

for i in range(len(x)):
    if y[i]==1 and dep_count>0:
        x_train.append(x[i])
        y_train.append(y[i])
        dep_count-=1
    elif y[i]==0 and nor_count>0:
        x_train.append(x[i])
        y_train.append(y[i])
        nor_count-=1
    else:
        x_test.append(x[i])
        y_test.append(y[i])

# print(len(x_train),type(x_train[0]))
np.save("x_train.npy",np.array(x_train,dtype=object))
np.save("y_train.npy",np.array(y_train,dtype=object))
np.save("x_test.npy",np.array(x_test,dtype=object))
np.save("y_test.npy",np.array(y_test,dtype=object))
