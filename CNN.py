from keras.models import Sequential
from keras.layers import Convolution2D  # for 2d images
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

cnn = Sequential()

# step 1: convolution
# slide feature detectors ("filters") along image
# results feature maps that form convolutional layer
cnn.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))  # 32, 3x3 filters

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

