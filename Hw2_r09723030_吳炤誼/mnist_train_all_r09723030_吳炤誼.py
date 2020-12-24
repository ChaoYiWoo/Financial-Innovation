# 插入所需套件
from sklearn.metrics import confusion_matrix
import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam

# lstm資料前處理
def lstm_preprocess(x_train, x_test, y_train, y_test, n_step, n_input, n_classes):
    # 把要訓練與測試的資料變成(n_step x n_input)
    x_train = x_train.reshape(-1, n_step, n_input)
    x_test = x_test.reshape(-1, n_step, n_input)
    # 把數據變成32 bit
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # (習慣:normalize)除255提升模型辨識力
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
    # 使用softmax fn將Y轉為機率值
    model.add(Activation('softmax'))
    return model

# cnn資料前處理
# 同lstm
def cnn_preprocess(x_train, x_test, y_train, y_test):
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, x_test, y_train, y_test)

# cnn model
def cnn_model():
    model = Sequential()
    # 二維捲積層(用5x5去捲,輸出28x28),超過的部份補零(same fn),用忽略負值的方式(relu fn)計算
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    # 池化層(取最大值來簡化)
    model.add(MaxPool2D(strides=2))
    # 再捲一次 & 池化
    model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    # 攤平維度 
    model.add(Flatten())
    # 疊三層
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def trainning(model, x_train, y_train, x_test, y_test, 
              learning_rate, training_iters, batch_size):
    # 學習速度(太大會一直波動，太小會浪費時間)
    adam = Adam(lr=learning_rate)
    model.summary()
    # 選擇優化函數,損失函數,衡量方式
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # 訓練模型
    model.fit(x_train, y_train,
              batch_size=batch_size, epochs=training_iters,
              verbose=1, validation_data=(x_test, y_test))

def print_confusion_result(x_train, x_test, y_train, y_test, model):
    # 得出預測值
    train_pred = model.predict_classes(x_train)
    test_pred = model.predict_classes(x_test)
    
    # 實際值
    train_label = y_train
    test_label =  y_test
    
    # 比較兩者，以confusion_matrix呈現(10x10) 越集中在對角線，越準確
    train_result_cm = confusion_matrix(train_label, train_pred, labels=range(10))
    test_result_cm = confusion_matrix(test_label, test_pred, labels=range(10))
    print(train_result_cm, '\n'*2, test_result_cm)

def mnist_lstm_main():
    # 給機器學的參數
    # adam預設為0.001
    learning_rate = 0.001
    # 迭代次數
    training_iters = 1
    # 每次樣本數
    batch_size = 128

    # 模型參數(層、步數、隱藏值、分成幾類)
    n_input = 28
    n_step = 28
    n_hidden = 256
    n_classes = 10

    #讀取資料，進行資料前處理
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test, y_train_o, y_test_o = lstm_preprocess(x_train, x_test, y_train, y_test, n_step, n_input, n_classes)

    #訓練lstm模型並印出結果
    model = lstm_model(n_input, n_step, n_hidden, n_classes)
    trainning(model, x_train, y_train_o, x_test, y_test_o, learning_rate, training_iters, batch_size)
    scores = model.evaluate(x_test, y_test_o, verbose=0)
    print('LSTM test accuracy:', scores[1])
    print_confusion_result(x_train, x_test, y_train, y_test, model)

def mnist_cnn_main():
    # 給機器學的參數
    learning_rate = 0.001
    training_iters = 1
    batch_size = 64

    #讀取資料，進行資料前處理
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test, y_train_o, y_test_o = cnn_preprocess(x_train, x_test, y_train, y_test)

    #訓練cnn模型並印出結果
    model = cnn_model()
    trainning(model, x_train, y_train_o, x_test, y_test_o, learning_rate, training_iters, batch_size)
    scores = model.evaluate(x_test, y_test_o, verbose=0)
    print('CNN test accuracy:', scores[1])
    print_confusion_result(x_train, x_test, y_train, y_test, model)


if __name__ == '__main__':
    mnist_lstm_main()
    
    mnist_cnn_main()