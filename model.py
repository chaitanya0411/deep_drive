'''
Our Final CNN model which gave the best results
after trying out many models 
'''

from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K

def build_cnn(image_size=None,weights_path=None):

    image_size = image_size or (128, 128)

    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3, )

    img_input = Input(input_shape)

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)
   
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    y = Flatten()(x)
    y = Dense(4096, activation='relu')(y)
    y = Dropout(.75)(y)
    y = Dense(2048, activation='relu')(y)
    y = Dropout(.75)(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(.75)(y)
    y = Dense(512, activation='relu')(y)
    y = Dropout(.75)(y)
    y = Dense(1)(y)

    model = Model(input=img_input, output=y)
    model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')

    if weights_path:
        model.load_weights(weights_path)

    return model
