import numpy as np
data1 = np.load('npy_files/image_fold_1.npy')
print(data1.shape)
data2 = np.load('npy_files/image_fold_2.npy')
print(data2.shape)
data3 = np.load('npy_files/image_fold_3.npy')
print(data3.shape)
data4 = np.load('npy_files/image_fold_4.npy')
print(data4.shape)

X_train = np.vstack([data1, data2, data3, data4])
print("X train shpe: ", X_train.shape)
np.save('X_train.npy', X_train)


data1 = np.load('npy_files/label_fold_1.npy')
print(data1.shape)
data2 = np.load('npy_files/label_fold_2.npy')
print(data2.shape)
data3 = np.load('npy_files/label_fold_3.npy')
print(data3.shape)
data4 = np.load('npy_files/label_fold_4.npy')
print(data4.shape)

Y_train = np.vstack([data1, data2, data3, data4])
print("Y train shpe: ", Y_train.shape)
np.save('Y_train.npy', Y_train)
