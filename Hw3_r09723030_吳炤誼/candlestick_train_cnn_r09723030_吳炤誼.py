# 插入所需套件
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPool2D


def load_pkl(pkl_name):
    # 讀取pkl檔
    with open(pkl_name, 'rb') as f:
        data = pickle.load(f)
    return data

# cnn model
def get_cnn_model(params):
    model = Sequential()
    # 二維捲積層(用5x5去捲,輸出10x10),超過的部份補零(same fn),用忽略負值的方式(relu fn)計算
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(10, 10, 4)))
    # 再捲一次
    model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    # 攤平維度 
    model.add(Flatten())
    # 疊三層(前兩層以忽略負值的方式算,最後一層用機率的方式算)
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    return model

# 訓練model
def train_model(params, data):
    model = get_cnn_model(params)
    # 選擇優化函數,損失函數,衡量方式
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    hist = model.fit(x=data['train_gaf'], y=data['train_label_arr'],
                     batch_size=params['batch_size'], epochs=params['epochs'], verbose=2)
    return (model, hist)

#印出結果
def print_result(data, model):
    # 得到訓練的值
    train_pred = model.predict_classes(data['train_gaf'])
    test_pred = model.predict_classes(data['test_gaf'])
    # 實際值
    train_label = data['train_label'][:, 0]
    test_label = data['test_label'][:, 0]
    # 比較兩者，以confusion_matrix呈現(9x9) 越集中在對角線，越準確
    train_result_cm = confusion_matrix(train_label, train_pred, labels=range(9))
    test_result_cm = confusion_matrix(test_label, test_pred, labels=range(9))
    print(train_result_cm, '\n'*2, test_result_cm)

if __name__ == "__main__":
    PARAMS = {}

    PARAMS['pkl_name'] = 'C:/Users/ben82/ipython_notebook_workplace/label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl'
    # 分類
    PARAMS['classes'] = 9
    # 學習速度(設為0.005準確度比0.01高)
    PARAMS['lr'] = 0.005
    # 迭代次數
    PARAMS['epochs'] = 20
    # 每次處理樣本數
    PARAMS['batch_size'] = 32
    #(設0.005,20,32)準確度提高
    PARAMS['optimizer'] = optimizers.SGD(lr=PARAMS['lr'])

    # ---------------------------------------------------------
    # 讀檔
    data = load_pkl(PARAMS['pkl_name'])
    # train cnn model
    model, hist = train_model(PARAMS, data)
    # train & test result
    scores = model.evaluate(data['test_gaf'], data['test_label_arr'], verbose=0)
    print('CNN test accuracy:', scores[1])
    print_result(data, model)