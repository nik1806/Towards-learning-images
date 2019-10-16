# from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import np_utils
import cv2
import numpy as np

#model creation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD


# (x_train, y_train), (x_test, y_test) = mnist.load_data()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# d_len, img_row, img_col = x_train.shape

# visualizing single data
# r_num = np.random.randint(0, x_len)
# img = x_train[r_num]
# cv2.imshow("random image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# x_train = x_train.reshape(d_len, img_row, img_col, 1)
# x_test = x_test.reshape(x_test.shape[0], img_row, img_col, 1)

# store the shape of a single image 
# input_shape = (img_row, img_col, 1)
input_shape = x_train.shape[1:]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255

#one hot encoder
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

# model
model = Sequential()
model.add(Conv2D(32, 
            kernel_size=(3,3), 
            activation='relu', 
            input_shape=input_shape))
model.add(Conv2D(64,
            kernel_size=(3,3),
            activation='relu'
            ))
model.add( MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
            optimizer = SGD(0.01),
            metrics=['accuracy'])

# print(model.summary())

# training model
batch_size = 32
epochs = 5

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy', score[1])

import matplotlib.pyplot as plt

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

# line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
# line2 = plt.plot(epochs, loss_values, label='Training Loss')
# plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
# plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
# plt.xlabel('Epochs') 
# plt.ylabel('Loss')
# plt.grid(True)
# plt.legend()
# plt.show()

model.save('mnist_simple_cnn_5_epoch.h5')

from keras.models import load_model
classifier = load_model('mnist_simple_cnn_5_epoch.h5')