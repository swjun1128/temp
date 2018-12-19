# -*- coding: utf-8 -*-
import keras
from keras.models import load_model
from keras.models import model_from_json
import h5py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
import csv
from keras.datasets import mnist

def HDF5_structure(data):
    root=data.keys()
    final_path=[]
    data_path=[]
    while True:
        if len(root)==0:
            break
        else:
            for item in root:
                if isinstance(data[item],h5py._hl.dataset.Dataset) or len(data[item].items())==0:
                    root.remove(item)
                    final_path.append(item)
                    if isinstance(data[item],h5py._hl.dataset.Dataset):
                        data_path.append(item)
                else:
                    for sub_item in data[item].items():
                        root.append(os.path.join(item,sub_item[0]))
                    root.remove(item)
    return data_path


def gaussian_fuzz(model,random_ratio=0.01):
    #期望是权重值，方差是w平方，不区分'kernel' or 'bias'都当做权重值处理
    #选取1%的权重值增加高斯噪声
    json_string=model.to_json()
    model.save_weights('my_model_weight.h5')
    
    with h5py.File('my_model_weight.h5', 'r+') as data:
        data_path=HDF5_structure(data)
        lst=[]
        for path in data_path:
            print data[path].shape
            if len(data[path].shape)==4:
                a,b,c,d=data[path].shape
                lst.extend([(path,a_index,b_index,c_index,d_index) for a_index in range(a) for b_index in range(b) for c_index in range(c) for d_index in range(d)])
            if len(data[path].shape)==2:
                row,col=data[path].shape
                lst.extend([(path,i,j) for i in range(row) for j in range(col)])
            else:
                row=data[path].shape[0]
                lst.extend([(path,i) for i in range(row)])
        random_choice=np.random.choice(range(len(lst)),replace=False,size=int(random_ratio*len(lst)))
        lst_random=np.array(lst)[[random_choice]]
    
        for path in lst_random:
            if len(path)==3:
                arr=data[path[0]][int(path[1])].copy()
                weight = arr[int(path[2])]
                #加上以权重值自身为均值和标准差的正态分布噪声
                arr[int(path[2])]+= np.random.normal(weight,abs(weight))
                data[path[0]][int(path[1])]=arr
            elif len(path)==2:
                arr=data[path[0]][int(path[1])]
                weight = arr
                 #加上以权重值自身为均值和标准差的正态分布噪声
                arr+= np.random.normal(weight,abs(weight))
                data[path[0]][int(path[1])]=arr
            elif len(path)==5:
                arr=data[path[0]][int(path[1])][int(path[2])][int(path[3])].copy()
                weight = arr[int(path[4])]
                 #加上以权重值自身为均值和标准差的正态分布噪声
                arr[int(path[4])]+= np.random.normal(weight,abs(weight))
                data[path[0]][int(path[1])][int(path[2])][int(path[3])]=arr
            
    model_change = model_from_json(json_string)
    model_change.load_weights('my_model_weight.h5')
    return model_change

def accuracy_mnist(model,mnist):
    '''
    model: DNN_model
    return : acc of mnist
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 输入数据为 mnist 数据集
    x_test = x_test.astype('float32').reshape(-1,28,28,1)
    x_test = x_test / 255

    y_test = keras.utils.to_categorical(y_test, 10)
    score = model.evaluate(x_test, y_test)
    return score[1]

def accuracy_cifar(model):
    #model: CNN_model
    #return : acc of cifar
    (_, _), (X_test, Y_test) = cifar10.load_data()
    X_test=X_test.astype('float32')
    X_test/=255
    pred=model.predict(X_test)
    pred=list(map(lambda x:np.argmax(x),pred))
    test_label=list(map(lambda x:np.argmax(x),pd.get_dummies(Y_test.reshape(-1)).values))
    return accuracy_score(test_label,pred)


if __name__=='__main__':
    model_path='../ModelC_raw.hdf5'
    model=load_model(model_path)
    score = accuracy_cifar(model)
    print('Origin Test accuracy: %.4f'% score)
    acc =[]
    for i in range(20):
        model_change = gaussian_fuzz(model,random_ratio=0.01)
        model_change.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        #print 'Mutated Test accuracy: ',accuracy_cifar(model_change)
        acc.append(accuracy_cifar(model_change))
    print acc