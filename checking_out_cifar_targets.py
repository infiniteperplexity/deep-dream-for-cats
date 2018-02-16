from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

from keras.applications import inception_v3