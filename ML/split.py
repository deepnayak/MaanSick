import numpy as np
import random
from sklearn.model_selection import train_test_split
from crop import crop
def split(file,interpolate):
    if interpolate:
        file = crop(file)
    fa_and_pd_new = np.load(file, allow_pickle=True)
    random.shuffle(fa_and_pd_new)
    x = [i[0] for i in fa_and_pd_new]
    y = [i[1] for i in fa_and_pd_new]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    np.save("x_train.npy",x_train)
    np.save("x_test.npy",x_test)
    np.save("y_train.npy",y_train)
    np.save("y_test.npy",y_test)