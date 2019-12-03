from keras.models import Sequential
from keras.layers import Convolution2D  # for 2d images
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

cnn = Sequential()

rgb = 64

# step 1: convolution
# slide feature detectors ("filters") along image
# results feature maps that form convolutional layer
cnn.add(Convolution2D(32, 3, 3, input_shape=(rgb, rgb, 3), activation='relu'))  # 32, 3x3 filters

# step 2: pooling
cnn.add(MaxPool2D(pool_size=(2, 2)))

# step 3: flatten
# this vector will be the input of a future ann
cnn.add(Flatten())

# step 4: full connection
cnn.add(Dense(output_dim=128, activation='relu'))  # add hidden layers
cnn.add(Dense(output_dim=1, activation='sigmoid'))  # sigmoid for binary output

# compile cnn
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# image augmentation - prevent overfitting
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(rgb, rgb),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(rgb, rgb),
        batch_size=32,
        class_mode='binary')

cnn.fit_generator(
        train_set,
        steps_per_epoch=8000, # we have 8k images in our training set
        epochs=10,
        validation_data=test_set,
        validation_steps=2000)

print(cnn.summary())

cnn.save('CatDogModel.h5')
