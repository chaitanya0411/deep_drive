'''
Trains the model, runs validation and testing.
Last 1600 images of the dataset are used for testing and validation.
'''

from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')

import argparse
import os

from data_generator import *
from model import *
import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint

def main():

    parser = argparse.ArgumentParser(description="Deep Drive model")
    parser.add_argument('--dataset', type=str, help='dataset folder with csv and image folders')
    parser.add_argument('--resized-image-height', type=int, help='image resizing')
    parser.add_argument('--resized-image-width', type=int, help='image resizing')
    parser.add_argument('--nb-epoch', type=int, help='# of training epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    args = parser.parse_args()

    dataset_path = args.dataset
    image_size = (args.resized_image_height, args.resized_image_width)
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    weights_path = None

    steering_log = path.join(dataset_path, 'steering.csv')
    image_log = path.join(dataset_path, 'camera.csv')
    camera_images = dataset_path

    model_builders = {'cnn': (build_cnn, normalize_input, exact_output)}

    model_builder, input_processor, output_processor = model_builders['cnn']
    model = model_builder(image_size,weights_path)

    training_data_generator = data_generator(steering_log=steering_log,
                                             image_log=image_log,
                                             image_folder=camera_images,
                                             gen_type='train',
                                             batch_size=batch_size,
                                             image_size=image_size,
                                             timestamp_start=None,
                                             timestamp_end=None,
                                             shuffle=True,
                                             preprocess_input=input_processor,
                                             preprocess_output=output_processor)


    validation_data_generator = data_generator(steering_log=steering_log,
                                               image_log=image_log,
                                               image_folder=camera_images,
                                               gen_type='val',
                                               batch_size=32,
                                               image_size=image_size,
                                               timestamp_start=14774405347-1600,
                                               timestamp_end=14774405347,
                                               shuffle=False,
                                               preprocess_input=input_processor,
                                               preprocess_output=output_processor)

    callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=2),
                ModelCheckpoint(filepath=os.path.join('weights.hdf5'),
                monitor='val_loss', verbose=2, save_best_only=True)]

    model.fit_generator(training_data_generator, samples_per_epoch=1,
                        nb_epoch=nb_epoch,verbose=2,
                        callbacks=callbacks, validation_data=validation_data_generator,
                        nb_val_samples=1600)

    print('Model trained successfully')

    test_data_generator = data_generator(steering_log=steering_log,
                                         image_log=image_log,
                                         image_folder=camera_images,
                                         gen_type='test',
                                         batch_size=1600,
                                         image_size=image_size,
                                         timestamp_start=14774405347-1600,
                                         timestamp_end=14774405347,
                                         shuffle=False,
                                         preprocess_input=input_processor,
                                         preprocess_output=output_processor)
       

    test_x, test_y = test_data_generator.next()

    print('test data shape:', test_x.shape, test_y.shape)

    yhat = model.predict(test_x)

    df_test = pd.read_csv('output1.csv',usecols=['frame_id','steering_angle','pred'],
                          index_col = None)
    df_test['steering_angle'] = test_y
    df_test['pred'] = yhat
    df_test.to_csv('predicted_angles.csv')

    sq = 0
    mse = 0
    for j in range(test_y.shape[0]):
        sqd = ((yhat[j]-test_y[j])**2)
        sq = sq + sqd
    print(sq)
    mse = sq/1600
    rmse = np.sqrt(mse)
    print("model evaluated RMSE:", rmse)

if __name__ == '__main__':
    main()
