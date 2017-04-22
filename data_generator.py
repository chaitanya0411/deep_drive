''' 
center camera being always chosen for training, validation
and testing
'''

from __future__ import print_function

import numpy as np
import random

from collections import defaultdict
from os import path
from scipy.misc import imread, imresize
from scipy import ndimage
from keras import backend as K

def read_steerings(steering_log, time_scale):

    steerings = defaultdict(list)
    speeds = defaultdict(list)

    with open(steering_log) as f:
        for line in f.readlines()[1:]:
            fields = line.split(",")
            nanosecond, angle, speed = int(fields[0]), float(fields[1]), float(fields[3])
            timestamp = nanosecond / time_scale
            steerings[timestamp].append(angle)
            speeds[timestamp].append(speed)
    return steerings, speeds

def read_image_stamps(image_log, time_scale):
    camera = 'center'
    timestamps = defaultdict(list)
    with open(image_log) as f:
        for line in f.readlines()[1:]:
            if camera not in line:
                continue
            fields = line.split(",")
            nanosecond = int(fields[0])
            timestamp = nanosecond / time_scale
            timestamps[timestamp].append(nanosecond)
    return timestamps


def read_images(image_folder, ids, image_size):

    prefix = path.join(image_folder, 'center')

    imgs = []

    for id in ids:
        img = imread(path.join(prefix, '%d.jpg' % id))

        crop_img = img[200:,:]

        img = imresize(crop_img, size=image_size)

        imgs.append(img)

    img_block = np.stack(imgs, axis=0)

    if K.image_dim_ordering() == 'th':
        img_block = np.transpose(img_block, axes = (0, 3, 1, 2))

    return img_block

# data augmentation
def read_images_augment(image_folder, ids, image_size):

    prefix = path.join(image_folder, 'center')
    imgs = []
    j = 0

    rand_no = random.randint(1, 4)

    #flip image
    if rand_no == 1:
        for id in ids:
            img = imread(path.join(prefix, '%d.jpg' % id))

            # Flip image
            img = np.fliplr(img)

            # Cropping
            crop_img = img[200:,:]

            # Resizing
            img = imresize(crop_img, size=image_size)

            imgs.append(img)
    # rotate image by a small amount
    elif rand_no == 2:
        for id in ids:
            img = imread(path.join(prefix, '%d.jpg' % id))

            crop_img = img[200:, :]

            img = imresize(crop_img, size=image_size)

            rotate = random.uniform(-1, 1)
            img = ndimage.rotate(img, rotate, reshape=False)

            imgs.append(img)
    #image blurring
    elif rand_no == 3:
        for id in ids:
            img = imread(path.join(prefix, '%d.jpg' % id))

            crop_img = img[200:, :]

            img = imresize(crop_img, size=image_size)

            #img = misc.face(gray=True)

            img = ndimage.gaussian_filter(img, sigma=3)

            imgs.append(img)
    else:
        # image sharpening
        for id in ids:
            img = imread(path.join(prefix, '%d.jpg' % id))

            crop_img = img[200:, :]

            img = imresize(crop_img, size=image_size)

            filter_blurred_f = ndimage.gaussian_filter(img, 1)
            alpha = 30
            img = img + alpha * (img - filter_blurred_f)

            imgs.append(img)


    img_block = np.stack(imgs, axis=0)

    if K.image_dim_ordering() == 'th':
        img_block = np.transpose(img_block, axes = (0, 3, 1, 2))

    return img_block


def normalize_input(x):
    return x / 255.


def exact_output(y):
    return y


def preprocess_input_InceptionV3(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def data_generator(steering_log, image_log, image_folder, gen_type='train',
                   batch_size=32, time_factor=10, image_size=0.5,
                   timestamp_start=None, timestamp_end=None, shuffle=True,
                   preprocess_input=normalize_input,
                   preprocess_output=exact_output):

    minmax = lambda xs: (min(xs), max(xs))
    time_scale = int(1e9) / time_factor

    if gen_type == 'train':
        image_stamps = read_image_stamps(image_log, time_scale)
    else:
        image_stamps = read_image_stamps(image_log, time_scale)

    steerings, speeds = read_steerings(steering_log, time_scale)

    print('timestamp range for all steerings: %d, %d' % minmax(steerings.keys()))
    print('timestamp range for all images: %d, %d' % minmax(image_stamps.keys()))
    print('min and max # of steerings per time unit: %d, %d' % minmax(map(len, steerings.values())))
    print('min and max # of images per time unit: %d, %d' % minmax(map(len, image_stamps.values())))

    start = max(min(steerings.keys()), min(image_stamps.keys()))

    if timestamp_start:
        start = max(start, timestamp_start)

    end = min(max(steerings.keys()), max(image_stamps.keys()))

    if timestamp_end:
        end = min(end, timestamp_end)

    print("sampling data from timestamp %d to %d" % (start, end))

    i = start
    x_buffer, y_buffer, buffer_size = [], [], 0

    while True:
        if i > end:
            i = start

        rand_no = random.randint(1, 3)

        if steerings[i] and image_stamps[i]:

            if rand_no == 1 or rand_no == 2:
                images = read_images(image_folder, image_stamps[i], image_size)
            else:
                images = read_images_augment(image_folder, image_stamps[i], image_size)

            angle = np.repeat([np.mean(steerings[i])], images.shape[0])

            x_buffer.append(images)
            y_buffer.append(angle)
            buffer_size += images.shape[0]

            print(buffer_size)

            if buffer_size >= batch_size:
                index = range(buffer_size)

                if gen_type == 'train':
                    np.random.shuffle(index)

                x = np.concatenate(x_buffer, axis=0)[index[:batch_size], ...]
                y = np.concatenate(y_buffer, axis=0)[index[:batch_size], ...]

                x_buffer, y_buffer, buffer_size = [], [], 0

                yield preprocess_input(x.astype(np.float32)), preprocess_output(y)

        if shuffle:
            i = np.random.randint(start, end)
        else:
            i += 1
