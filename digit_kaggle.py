# -*- coding: utf-8 -*-
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
# -*- coding: utf-8 -*-
import numpy as np
import csv

def LoadTraincsvfile():
    tmp=np.loadtxt('/Users/itwork/Desktop/train_set/big-mnist/train.csv',dtype=np.str,delimiter=',')
    x_data=tmp[1:,1:].astype(np.float)
    y_data=tmp[1:,0].astype(np.float)
    return x_data,y_data
x_data,y_data=LoadTraincsvfile()
path = './mnist.npz'
f = np.load(path)
x_train1, y_train1 = f['x_train'], f['y_train']
x_test1, y_test1 = f['x_test'], f['y_test']
f.close()
print (x_data[0:41000,:].shape)
print (x_train1.shape)
x_train1 = x_train1.reshape(60000, 784).astype('float32')
x_train=np.concatenate((x_data[0:42000,:],x_train1),axis=0)
y_train=np.concatenate((y_data[0:42000],y_train1),axis=0)
x_test=x_data[41000:,:]
y_test=y_data[41000:]
print (x_train.shape)
print (y_train.shape)
x_train /= 255
x_test /= 255
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

model = Sequential()

model.add(Dense(200, activation='relu', input_dim=784))
model.add(Dense(120, activation='relu'))
model.add(Dense(10, activation='softmax'))

rms = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=50,
          batch_size=256)
score = model.evaluate(x_test, y_test, batch_size=4096)
print(model.metrics_names)
print(score)

#预测部分
is_predict=1
if(is_predict):
    tmp = np.loadtxt('/Users/itwork/Desktop/train_set/big-mnist/test.csv', dtype=np.str, delimiter=',')
    x_predict = tmp[1:, :].astype(np.float)
    x_predict /= 255

    classes = model.predict_classes(x_predict)
    resultlist = []
    imgid = 1
    for i in classes:
        res_data = {'ImageId': imgid, 'Label': i}
        resultlist.append(res_data)
        imgid = imgid + 1

    headers = ['ImageId', 'Label']
    with open('result.csv', 'w', newline='') as f:
        # 标头在这里传入，作为第一行数据
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        for row in resultlist:
            writer.writerow(row)

