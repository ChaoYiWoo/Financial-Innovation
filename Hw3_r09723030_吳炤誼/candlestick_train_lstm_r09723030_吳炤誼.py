# 插入所需套件
from sklearn.metrics import confusion_matrix
import pickle
import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam


def load_pkl(pkl_name):
    # 讀candlestick檔
    with open(pkl_name, 'rb') as f:
        data = pickle.load(f)
    return data

# lstm資料前處理
def lstm_preprocess(x_train, x_test, y_train, y_test, n_step, n_input, n_classes):
    # 把要訓練與測試的資料變成(n_step x n_input)
    x_train = x_train.reshape(-1, n_step, n_input)
    x_test = x_test.reshape(-1, n_step, n_input)
    # 把數據變成32 bit
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # (:normalize)除255提升模型辨識力
    x_train /= 255
    x_test /= 255
    # 轉成特定的處理格式(one hot)
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    return (x_train, x_test, y_train, y_test)

# lstm model
def lstm_model(n_input, n_step, n_hidden, n_classes):
    model = Sequential()
    # 加入隱藏值
    model.add(LSTM(n_hidden, batch_input_shape=(None, n_step, n_input), unroll=True))
    # 輸出層
    model.add(Dense(n_classes))
    # 使用Activation中softmax fn將Y轉為機率值
    model.add(Activation('softmax'))
    return model

# 訓練lstm model
def train_lstm(model, x_train, y_train, x_test, y_test, 
        learning_rate, training_iters, batch_size):
    # 學習速度(太大會在兩側波動，太小會浪費時間)
    adam = Adam(lr=learning_rate)
    model.summary()
    # 選擇優化函數,損失函數,衡量方式
    model.compile(optimizer=adam,
        loss='categorical_crossentropy', metrics=['accuracy'])
    # 訓練模型
    model.fit(x_train, y_train,
        batch_size=batch_size, epochs=training_iters,
        verbose=1, validation_data=(x_test, y_test))

def print_result(data, x_train, x_test, model):
    # 得出預測值
    train_pred = model.predict_classes(x_train)
    test_pred = model.predict_classes(x_test)
    # 實際值
    train_label = data['train_label'][:, 0]
    test_label = data['test_label'][:, 0]
    # 比較兩者，以confusion_matrix呈現(9x9) 越集中在對角線，越準確
    train_result_cm = confusion_matrix(train_label, train_pred, labels=range(9))
    test_result_cm = confusion_matrix(test_label, test_pred, labels=range(9))
    print(train_result_cm, '\n'*2, test_result_cm)

def mnist_lstm_main():
    # 給機器學的參數
    # adam學習速度預設為0.001
    learning_rate = 0.005
    #迭代次數
    training_iters = 20
    # 學習速度調至0.005&迭代20次得到比原本較高的準確率)
    # 每次樣本數
    batch_size = 128

    # 模型參數(層、步數、隱藏值(特徵數)、分成幾類)
    n_input = 40
    n_step = 10
    n_hidden = 256
    n_classes = 10
   
    #讀取資料，進行資料前處理
    data = load_pkl('C:/Users/ben82/ipython_notebook_workplace/label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl')
    x_train, y_train, x_test, y_test = data['train_gaf'], data['train_label'][:, 0], data['test_gaf'], data['test_label'][:, 0]
    x_train, x_test, y_train, y_test = lstm_preprocess(x_train, x_test, y_train, y_test, n_step, n_input, n_classes)

    #訓練lstm模型並印出結果
    model = lstm_model(n_input, n_step, n_hidden, n_classes)
    train_lstm(model, x_train, y_train, x_test, y_test, learning_rate, 
               training_iters, batch_size)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('LSTM test accuracy:', scores[1])
    print_result(data, x_train, x_test, model)

if __name__ == '__main__':
    mnist_lstm_main()