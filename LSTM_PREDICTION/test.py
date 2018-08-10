import keras
from keras.utils import plot_model

def LSTM(num_in=5, num_out=2):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(100, kernel_initializer='normal', input_shape=(num_in, num_out), activation='relu'))
    model.add(keras.layers.LSTM(100, input_shape=(num_in, num_out)))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model

model = LSTM()
plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True)