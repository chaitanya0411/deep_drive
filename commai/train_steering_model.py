#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.models import Model, Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras import backend as K


from server import client_generator


def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    X, Y, _ = tup
    Y = Y[:, -1]
    if X.shape[1] == 1:  # no temporal context
      X = X[:, -1]
    yield X, Y

# replaced our best performing model here

def get_model(time_len=1):
  ch, row, col = 3, 160, 320  # camera format

  input_shape = (ch, row, col)


  img_input = Input(input_shape)

  x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(img_input)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)
  x = Dropout(0.25)(x)

  x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(img_input)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)
  x = Dropout(0.25)(x)

  x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
  x = Dropout(0.25)(x)

  y = Flatten()(x)
  y = Dense(4096, activation='relu')(y)
  y = Dropout(.5)(y)
  y = Dense(2048, activation='relu')(y)
  y = Dropout(.5)(y)
  y = Dense(1024, activation='relu')(y)
  y = Dropout(.5)(y)
  y = Dense(512, activation='relu')(y)
  y = Dropout(.5)(y)
  y = Dense(1)(y)

  model = Model(input=img_input, output=y)
  model.compile(optimizer=Adam(lr=1e-4), loss='mse')

  return model


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=500, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  model = get_model()
  model.fit_generator(
    gen(20, args.host, port=args.port),
    samples_per_epoch=10000,
    nb_epoch=args.epoch,
    validation_data=gen(20, args.host, port=args.val_port),
    nb_val_samples=1000
  )
  print("Saving model weights and configuration file.")

  if not os.path.exists("./outputs/steering_model"):
      os.makedirs("./outputs/steering_model")

  model.save_weights("./outputs/steering_model/steering_angle.keras", True)
  with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
