from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import AveragePooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import multi_gpu_model

import numpy as np
from skimage.transform import resize
from skimage import io

from os import listdir
from os.path import join
import collections
import keras.backend as K
import tools.image_gen_extended as T
import multiprocessing as mp
from math import ceil
import sys

num_gpus = 4
batch_size = 64 * num_gpus
data_path = '/data/cephfs/punim0847/'
num_processes = 24


# Load part of dataset images (since I do not have enough memory) and resize to meet minimum width and height pixel size
def load_images(root, img_num_limit, min_side=299):
    all_imgs = []
    all_classes = []
    resize_count = 0
    invalid_count = 0
    for i, subdir in enumerate(listdir(root)):
        imgs = listdir(join(root, subdir))
        class_ix = class_to_ix[subdir]
        print(i, class_ix, subdir)
        img_count = 0
        for img_name in imgs:
            img_count = img_count + 1
            if img_count > img_num_limit:
                break

            # img_arr = img.imread(join(root, subdir, img_name))
            # piexif.remove(join(root, subdir, img_name))
            img_arr = io.imread(join(root, subdir, img_name))
            img_arr_rs = img_arr
            try:
                w, h, _ = img_arr.shape
                if w < min_side:
                    wpercent = (min_side / float(w))
                    hsize = int((float(h) * float(wpercent)))
                    # print('new dims:', min_side, hsize)
                    img_arr_rs = resize(img_arr, (min_side, hsize))
                    resize_count += 1
                elif h < min_side:
                    hpercent = (min_side / float(h))
                    wsize = int((float(w) * float(hpercent)))
                    # print('new dims:', wsize, min_side)
                    img_arr_rs = resize(img_arr, (wsize, min_side))
                    resize_count += 1
                all_imgs.append(img_arr_rs)
                all_classes.append(class_ix)
            except:
                print('Skipping bad image: ', subdir, img_name)
                invalid_count += 1
    print(len(all_imgs), 'images loaded')
    print(resize_count, 'images resized')
    print(invalid_count, 'images skipped')
    return np.array(all_imgs), np.array(all_classes)


def generate_train(x_train, y_train, mp_pool):
    # this is the augmentation configuration we will use for training
    train_datagen = T.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        zoom_range=[.8, 1],
        channel_shift_range=30,
        fill_mode='reflect')
    train_datagen.config['random_crop_size'] = (299, 299)
    train_datagen.set_pipeline([T.random_transform, T.random_crop, T.preprocess_input])
    return train_datagen.flow(x_train, y_train, batch_size=batch_size, seed=3, pool=mp_pool)


def generate_test(x_test, y_test, mp_pool):
    test_datagen = T.ImageDataGenerator()
    test_datagen.config['random_crop_size'] = (299, 299)
    test_datagen.set_pipeline([T.random_transform, T.random_crop, T.preprocess_input])
    return test_datagen.flow(x_test, y_test, batch_size=batch_size, seed=11, pool=mp_pool)


def model_training(train, test, num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
    x = base_model.output
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(.4)(x)
    x = Flatten()(x)
    predictions = Dense(num_classes, kernel_initializer='glorot_uniform', kernel_regularizer=l2(.0005),
                        activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    opt = SGD(lr=.01, momentum=.9)
    parallel_model = multi_gpu_model(model, gpus=num_gpus)
    parallel_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=join(data_path, 'model4.{epoch:02d}-{val_loss:.2f}.hdf5'), verbose=1,
                                   save_best_only=True, save_weights_only=True)
    csv_logger = CSVLogger('model4-2.log')

    def schedule(epoch):
        if epoch < 15:
            return .01
        elif epoch < 28:
            return .002
        else:
            return .0004

    lr_scheduler = LearningRateScheduler(schedule)

    parallel_model.fit_generator(train,
                                 validation_data=test,
                                 validation_steps=ceil(X_test.shape[0] / batch_size),
                                 steps_per_epoch=ceil(X_train.shape[0] / batch_size),
                                 epochs=32,
                                 verbose=2,
                                 callbacks=[lr_scheduler, csv_logger, checkpointer])


if __name__ == '__main__':
    train_n = int(sys.argv[1])
    test_n = int(sys.argv[2])

    pool = mp.Pool(processes=num_processes)

    class_to_ix = {}
    ix_to_class = {}
    with open(join(data_path, 'meta/classes.txt'), 'r') as txt:
        classes = [l.strip() for l in txt.readlines()]
        class_to_ix = dict(zip(classes, range(len(classes))))
        ix_to_class = dict(zip(range(len(classes)), classes))
        class_to_ix = {v: k for k, v in ix_to_class.items()}
    sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))

    X_test, Y_test = load_images(join(data_path, 'food/test'), test_n, min_side=299)

    X_train, Y_train = load_images(join(data_path, 'food/train'), train_n, min_side=299)

    print('X_train shape', X_train.shape)
    print('y_train shape', Y_train.shape)
    print('X_test shape', X_test.shape)
    print('y_test shape', Y_test.shape)

    n_classes = 101
    y_train_cat = to_categorical(Y_train, num_classes=n_classes)
    y_test_cat = to_categorical(Y_test, num_classes=n_classes)

    train_generator = generate_train(X_train, y_train_cat, pool)
    test_generator = generate_test(X_test, y_test_cat, pool)

    K.clear_session()

    print("Start Training")

    model_training(train_generator, test_generator, n_classes)

