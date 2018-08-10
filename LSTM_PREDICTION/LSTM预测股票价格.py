import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import os
from keras.utils import plot_model


'---------------------------------------------'
# 设置运行环境

pd.set_option('display.width',450)
rlog = r'log\log_tmp'
if os.path.exists(rlog):
    tf.gfile.DeleteRecursively(rlog)

# 定义函数，从df中取指定的列
# 预测第二天的平均价格，所以对DataFrame进行相应的处理
def reformat_df(df):
    new_df = df[['open','high','low','close','volume','avg']]
    new_df['xavg'] = new_df['avg'].shift(-1)
    df.pop('avg')
    return new_df

# 定义函数，对数据样本进行分割
def train_test_split(df):
    row_count = df['open'].count()
    df_train = df.head(row_count-1)
    df_test = df.tail(1)

    x_train = df_train[['open','high','low','close','volume']].values
    y_train = df_train[['xavg']]
    x_test = df_test[['open','high','low','close','volume']].values
    return x_train,y_train,x_test

# 定义np_reshape函数，形成新的三维数组，以满足LSTM模型的需要
def np_reshape(ndarray):
    row_shape = ndarray.shape[0]
    column_shape = ndarray.shape[1]
    new_ndarray = ndarray.reshape(row_shape,column_shape,-1)
    return new_ndarray


'---------------------------------------------'
# 读取数据,并对数据进行分割

path = r'data/TDS2_sz50.csv'
df = pd.read_csv(path)
df = reformat_df(df)
x_train,y_train,x_test = train_test_split(df)
x_train = np_reshape(x_train)
x_test = np_reshape(x_test)

'---------------------------------------------'
# 建立LSTM网络模型
def LSTM(num_in=5,num_out=1):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(num_in*8,input_shape=(num_in,num_out)))
    model.add(keras.layers.Dense(100,input_shape=(num_in,1)))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
    return model

model = LSTM()
model.summary()

'---------------------------------------------'
# 对模型进行训练
tbcallback = keras.callbacks.TensorBoard(log_dir=rlog,write_images=True,write_graph=True,write_grads=True)
model.fit(x_train,y_train,epochs=500,batch_size=128,verbose=2,callbacks=[tbcallback])
model.save('LSTM.dat')
model.save_weights('LSTM_weights.dat')

'---------------------------------------------'
#对结果进行预测
y_pred = model.predict(x_test)
print(y_pred)





