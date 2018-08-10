import tensorflow as tf
import keras
from keras.utils import plot_model
import pandas as pd
import numpy as np
import glob
import time

'------------------------------------------'


# 定义np_reshape函数，形成新的三维数组，以满足LSTM模型的需要
def np_reshape(ndarray):
    row_shape = ndarray.shape[0]
    column_shape = ndarray.shape[1]
    new_ndarray = ndarray.reshape(row_shape, column_shape, -1)
    return new_ndarray


# 定义函数，对数据样本进行分割
def train_test_split(df):
    row_count = df['open'].count()
    df_train = df.head(row_count - 1)
    df_test = df.tail(1)

    raw_x_train = df_train[['open', 'high', 'low', 'close', 'volume']].values
    x_train_list = [df_train.loc[i - 30:i - 1, ['open', 'high', 'low', 'close', 'volume']].values.reshape(1, -1) for i
                    in
                    range(30, len(df_train.index.values))]
    x_train = np.row_stack(x_train_list)
    y_train_open = df_train.loc[30:, 'xopen'].values
    y_train_close = df_train.loc[30:, 'xclose'].values
    raw_x_test = df_test[['open', 'high', 'low', 'close', 'volume']].values
    x_test_df = df.tail(30)
    x_test = x_test_df[['open', 'high', 'low', 'close', 'volume']].values.reshape(1, -1)
    x_train = np_reshape(x_train)
    x_test = np_reshape(x_test)
    raw_x_train = np_reshape(raw_x_train)
    raw_x_test = np_reshape(raw_x_test)
    return x_train, y_train_open, y_train_close, x_test, raw_x_train, raw_x_test


'------------------------------------------'


# 制作时间计数器
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('执行时间：[%s seconds]' % (end_time - start_time))
        return res

    return wrapper


'------------------------------------------'


# 构建模型
def LSTM(num_in=150, num_out=1):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(num_in * 8, input_shape=(num_in, num_out)))  # 5列1层数据
    model.add(keras.layers.Dense(100, kernel_initializer='normal', input_shape=(num_in, num_out), activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model


model = LSTM()
model.summary()
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='log/log_tmp', write_graph=True, write_images=True,
                                                   write_grads=True)

'------------------------------------------'


# 定义训练过程函数
@timer
def train_open(model, x_train, y_train_open, x_test, raw_x_test, stock):
    model.fit(x_train, y_train_open, batch_size=160, epochs=30000, verbose=2)
    model.save(r'stock_model_data\%s_open.dat' % stock, overwrite=True)
    model.save_weights(r'stock_model_data\%s_open_weights.dat' % stock, overwrite=True)
    y_pred = model.predict(x_test)
    y_pred = y_pred.flatten()
    x_test = x_test[:, :, 0]
    new_df = pd.DataFrame()
    new_df['open'] = pd.Series(raw_x_test[:, 0])
    new_df['high'] = pd.Series(raw_x_test[:, 1])
    new_df['low'] = pd.Series(raw_x_test[:, 2])
    new_df['close'] = pd.Series(raw_x_test[:, 3])
    new_df['volume'] = pd.Series(raw_x_test[:, 4])
    new_df['open_next_day'] = pd.Series(y_pred)
    new_df.to_csv(r'stock_price_prediction_open\%s_open.csv' % stock)
    print('%s open is done' % stock)


@timer
def train_close(model, x_train, y_train_close, x_test, raw_x_test, stock):
    model.fit(x_train, y_train_close, batch_size=160, epochs=30000, verbose=2)
    model.save(r'stock_model_data\%s_close.dat' % stock, overwrite=True)
    model.save_weights(r'stock_model_data\%s_close_weights.dat' % stock, overwrite=True)
    y_pred = model.predict(x_test)
    y_pred = y_pred.flatten()
    x_test = x_test[:, :, 0]
    new_df = pd.DataFrame()
    # new_df[['open', 'high', 'low', 'close','volume']] = pd.DataFrame(x_test[:, 0:5])
    new_df['open'] = pd.Series(raw_x_test[:, 0])
    new_df['high'] = pd.Series(raw_x_test[:, 1])
    new_df['low'] = pd.Series(raw_x_test[:, 2])
    new_df['close'] = pd.Series(raw_x_test[:, 3])
    new_df['volume'] = pd.Series(raw_x_test[:, 4])
    new_df['close_next_day'] = pd.Series(y_pred)
    new_df.to_csv(r'stock_price_prediction_close\%s_close.csv' % stock)
    print('%s close is done' % stock)


'------------------------------------------'


# 训练模型并保存结果
@timer
def generate(model):
    stock_data_list = glob.glob(r'stock_data\*.csv')
    total_num = len(stock_data_list)
    count = 0
    for stock in stock_data_list:
        print('执行第%s/%s只股票完毕' % (count, total_num))
        stock_num = stock.lstrip(r'stock_data\\').rstrip('.csv')
        stock_df = pd.read_csv(stock)
        x_train, y_train_open, y_train_close, x_test, raw_x_train, raw_x_test = train_test_split(stock_df)
        train_open(model, x_train, y_train_open, x_test, raw_x_test, stock_num)
        train_close(model, x_train, y_train_close, x_test, raw_x_test, stock_num)
        count += 1


if __name__ == '__main__':
    generate(model)
