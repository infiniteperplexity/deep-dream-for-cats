from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

base_model = InceptionV3(weights="imagenet", include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(200, activation="softmax")(x)
model = Model(input=base_model.input, output=predictions)

for layer in base_model.layers: layer.trainable = False

model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss="categorical_crossentropy")

#so the next step confuses me...
# http://www.robots.ox.ac.uk/~vgg/data/pets/
# URL for CIFAR 100: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a