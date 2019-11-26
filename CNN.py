from keras.models import Sequential
from keras.layers import Convolution2D  # for 2d images
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

cnn = Sequential()

# step 1: convolution
# slide feature detectors ("filters") along image
# results feature maps that form convolutional layer
cnn.add(Convolution2D())