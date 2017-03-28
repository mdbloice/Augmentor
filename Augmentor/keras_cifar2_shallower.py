'''Trains a simple convnet on the MNIST dataset.



Gets to 99.25% test accuracy after 12 epochs

(there is still a lot of margin for parameter tuning).

16 seconds per epoch on a GRID K520 GPU.

'''



from __future__ import print_function

import numpy as np
from Operations import RandomNoise

from keras.keras.datasets import mnist
from keras.keras.models import Sequential
from keras.keras.datasets import cifar10
from keras.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.keras import optimizers
from keras.keras import losses
from keras.keras import utils
from keras.keras import backend as K

data_augmentation = True

batch_size = 1280

num_classes = 2

epochs = 12



# input image dimensions

img_rows, img_cols = 32, 32 # for minst 28x28

# the data, shuffled and split between train and test sets

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

first_class = y_train==0
second_class = y_train==1
together = np.array(first_class + second_class)

x_train = x_train[np.reshape(together, together.shape[0])]
y_train = y_train[np.reshape(together, together.shape[0])]
first_class = y_test==0
second_class = y_test==1
together = np.array(first_class + second_class)

x_test = x_test[np.reshape(together, together.shape[0])]
y_test = y_test[np.reshape(together, together.shape[0])]
if K.image_data_format() == 'channels_first':

    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols) # 1 for minst peste tot unde e 3

    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)

    input_shape = (3, img_rows, img_cols)

else:

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

    input_shape = (img_rows, img_cols, 3)



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices

y_train = utils.to_categorical(y_train, num_classes)

y_test = utils.to_categorical(y_test, num_classes)




model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=losses.categorical_crossentropy,

              optimizer=optimizers.Adadelta(),

              metrics=['accuracy'])
if not data_augmentation:

    print('Not using data augmentation.')


    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,

              verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])

    print('Test accuracy:', score[1])

else:
    print("Using Data Augmentation")
    new_x_train = np.zeros(([100000,32,32,3]))
    new_y_train = np.zeros(([100000,2]))
    rnoiseoperation = RandomNoise(0.9)
    for i in range(x_train.shape[0]):
        for j in range(0,10): # out of each image do 10 new ones
            new_x_train[i,:,:,:] = rnoiseoperation.perform_operation(x_train[i,:,:,:])
            new_y_train[i,:] = y_train[i,:]

    # at the end combine the datasets
    x_train = np.concatenate([x_train, new_x_train], axis=0)
    y_train = np.concatenate([y_train, new_y_train], axis=0)
    del new_y_train
    del new_x_train
    temp = list(zip(x_train, y_train))
    np.random.shuffle(temp)
    x_train, y_train = zip(*temp)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,

              verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])

    print('Test accuracy:', score[1])