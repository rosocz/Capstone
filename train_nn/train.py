import os
import numpy as np
import pandas as pd
import random
import cv2
import csv
import glob
from sklearn import model_selection
from skimage.transform import rescale

from keras import backend as K
from keras import models, optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, Activation, MaxPooling2D, Reshape, Input, concatenate
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

ROOT_PATH = './'
BATCH_SIZE = 4
EPOCHS = 60
NUM_CLASSES = 4

IMAGE_HEIGHT = 124
IMAGE_WIDTH = 64
IMAGE_CHANNEL = 3

AUGMENT_INDEX = 6

MODEL_FILE_NAME = './tl_classifier.h5'

# check for GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def create_labeled_list():
    with open('traffic_light_train.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(['path', 'class', 'color'])

        classes_distribution = [0, 0, 0, 0]
        for myclass, directory in enumerate(['NoTrafficLight', 'Red', 'Yellow', 'Green']):
            for filename in glob.glob('./training_data/real/{}/*.png'.format(directory)):
                filename = '/'.join(filename.split('\\'))
                mywriter.writerow([filename, myclass, directory])
                if (directory == 'Red'):
                    classes_distribution[0] += 1
                if (directory == 'Green'):
                    classes_distribution[1] += 1
                if (directory =="Yellow"):
                    classes_distribution[2] += 1
                if (directory == 'NoTrafficLight'):
                    classes_distribution[3] += 1

def analyse_data_distribution(data):
    distribution = data['color'].value_counts()
    # max_count = distribution.max()
    # print(max_count - distribution['Red'])

    return distribution


def random_brightness(image):
    # Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    # Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def zoom(image):
    zoom_pix = random.randint(0, 10)
    zoom_factor = 1 + (2*zoom_pix)/IMAGE_HEIGHT
    image = cv2.resize(image, None, fx=zoom_factor,
                       fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    top_crop = (image.shape[0] - IMAGE_HEIGHT)//2
    left_crop = (image.shape[1] - IMAGE_WIDTH)//2
    image = image[top_crop: top_crop+IMAGE_HEIGHT,
                  left_crop: left_crop+IMAGE_WIDTH]
    return image

def noise(image):
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image.shape]
    out[coords] = 0

    return out


# loads image
def get_image(index, data):
    # pair image and color clasiffication
    image = cv2.imread(os.path.join(data['path'].values[index].strip()))
    color = data['class'].values[index]

    return [image, color]

def augment(image):

    if (random.randint(0, 1)):
        image = random_brightness(image)

    if (random.randint(0, 1)):
        image = cv2.flip(image, 1)

    if (random.randint(0, 1)):
        image = zoom(image)

    if (random.randint(0, 1)):
        image = noise(image)

    return image

#normalize image canvas
def normalize_canvas_size(image):
    normalized_canvas = np.ndarray((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype=np.uint8)
    if (image.shape[0] > IMAGE_HEIGHT):
        coeff = image.shape[0] / IMAGE_HEIGHT
        image = rescale(image, 1.0 / coeff, mode='constant')

    if (image.shape[1] > IMAGE_WIDTH):
        coeff = image.shape[1] / IMAGE_WIDTH
        image = rescale(image, 1.0 / coeff, mode='constant')

    h, w = image.shape[:2]
    normalized_canvas[:h, :w] = image

    return normalized_canvas

# generator function to return images batchwise
def generator(data, has_augment=False):
    while True:
        # Randomize the indices to make an array
        indices_arr = np.random.permutation(data.count()[0])
        for batch in range(0, len(indices_arr), BATCH_SIZE):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + BATCH_SIZE)]

            # initializing the arrays, x_train and y_train
            x_train = np.empty(
                [0, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], dtype=np.float32)
            y_train = np.empty([0], dtype=np.int32)

            for i in current_batch:
                [image, color] = get_image(i, data)

                image = normalize_canvas_size(image)

                x_train = np.append(x_train, [image], axis=0)
                y_train = np.append(y_train, [color])

                if (has_augment):
                    distribution = analyse_data_distribution(data)
                    for i in range(0, int((((distribution.max() * AUGMENT_INDEX)-distribution[color])/distribution[color])/BATCH_SIZE)):
                        augmented_image = augment(image)
                        x_train = np.append(x_train, [augmented_image], axis=0)
                        y_train = np.append(y_train, [color])
            y_train = to_categorical(y_train, num_classes=NUM_CLASSES)

            yield (x_train, y_train)


def get_model():

    model = Sequential()

    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)
    model.add(Conv2D(72, 18, strides=2, input_shape=input_shape,padding='same', activation='relu'))
    model.add(Lambda(lambda x: x/127.5 - 1.))
    model.add(Conv2D(32, 16, strides=2, padding="same", activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(16, 16, strides=2, padding="same", activation='relu'))
    model.add(MaxPooling2D(2, 2))
#     model.add(Conv2D(16, 8, strides=2, padding="same", activation='relu'))
#     model.add(MaxPooling2D(2, 2))
#     model.add(Conv2D(32, 4, strides=2, padding="same", activation='relu'))
#     model.add(MaxPooling2D(2, 2))
#     model.add(Conv2D(32, 4, strides=2, padding="same", activation='relu'))
#     model.add(MaxPooling2D(2, 2))
#     model.add(Conv2D(32, 2, strides=1, padding="same", activation='relu'))
#     model.add(MaxPooling2D(2, 2))
#     model.add(Conv2D(64, 5, strides=(2, 2), padding="same", activation='relu'))
    model.add(Flatten())
    model.add(Dropout(.35))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(NUM_CLASSES))
    # model.add(Lambda(lambda x: (K.exp(x) + 1e-4) / (K.sum(K.exp(x)) + 1e-4)))
    model.add(Lambda(lambda x: K.tf.nn.softmax(x)))


    model.compile(optimizer=Adam(lr=5e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

if __name__ == "__main__":

    if not os.path.exists('./traffic_light_train.csv'):
        create_labeled_list()
        print('CSV file created successfully')
    else:
        print('CSV already present')

    data = pd.read_csv(os.path.join('./traffic_light_train.csv'))

    # Split data into random training and validation sets
    d_train, d_valid = model_selection.train_test_split(data, test_size=.2)

    train_gen = generator(d_train, True)
    validation_gen = generator(d_valid, False)

    model = get_model()

    # checkpoint to save best weights after each epoch based on the improvement in val_loss
    checkpoint = ModelCheckpoint(MODEL_FILE_NAME, monitor='val_loss', verbose=1,save_best_only=True, mode='min',save_weights_only=False)
    callbacks_list = [checkpoint] #,callback_each_epoch]

    print('Training started....')

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=len(d_train)//BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_gen,
        validation_steps=len(d_valid)//BATCH_SIZE,
        verbose=1,
        callbacks=callbacks_list
    )

    # print("Saving model..")
    # model.save("./tl_classifier_keras.h5")
    # print("Model Saved successfully!!")

    # Destroying the current TF graph to avoid clutter from old models / layers
    K.clear_session()
