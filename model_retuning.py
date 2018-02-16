### Load and format re-training images
import os
import numpy as np
import scipy
from keras.applications import inception_v3
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

WIDTH, HEIGHT = 299, 299 # image width and height from ImageNet
DENSE = 1024 # I think this is just a feature of Inception_V3
EPOCHS = 3 # isn't this quite small?
CLASSES = 12 # twelve breeds of cat
FROZEN = 172 # number of layers to freeze, should probably change
BATCHSIZE = 32 # is this arbitrary?
SPLIT = 0.2
VERBOSE = 2

# borrowed from the Keras Deep Dream implementation
def preprocess_image(image_path):
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

input_path = 'C:/Users/M543015/Desktop/GitHub/deeplearning/images/images/'
### Load cat images only and build list of breeds
cats = [name for name in os.listdir(input_path) if "jpg" in name and name[0].isupper()]
breeds = []
pairs = []
for cat in cats:
    breed = "_".join(cat.split("_")[:-1])
    if breed not in breeds:
        breeds.append(breed)

    pairs.append((cat,len(breeds)-1))

images, labels = zip(*[(preprocess_image(input_path+pair[0]),pair[1]) for pair in pairs])

### This was weirdly finicky about working as a function so I'll do it as a loop
resized = []
for img in images:
    resized.append(scipy.ndimage.zoom(img, (1, float(WIDTH)/img.shape[1], float(HEIGHT)/img.shape[2], 1), order=1))

X_train = np.vstack(resized)
## assuming here that I need one-hot encoding
y_train = to_categorical(np.vstack(labels))

# train only the new layers, so as to avoid disturbing the pretrained weights
base_model = InceptionV3(weights="imagenet", include_top=False)
output_layer = base_model.ouput
final_pooling = GlobalAveragePooling2D()(output_layer)
final_dense = Dense(DENSE, activation="relu")(final_pooling)
predictions = Dense(CLASSES,, activation="softmax")(final_dense)
model = (input=base_model.input, output=predictions)
for layer in base_model.layers:
	layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# data augmentation...I hope this does not mess up the preprocessing
myImageGenerator = ImageDataGenerator(
	rotation_range=30,
	width_shift_range=0.2,
	height_shift_range=0.2,
	zoom_range=0.2,
	shear_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest'
)

# fit model
transfer_history = model.fit_generator(
	myImageGenerator.flow(X_train, y_train, batch_size=BATCHSIZE),
	samples_per_epoch=X_train.shape[0],
	epochs=EPOCHS,
	verbose=VERBOSE,
	validation_split=SPLIT
)

# retrain chosen number of layers
for layer in model.layers[:FROZEN]:
	layer.trainable = False

for layer in model.layers[FROZEN:]:
	layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# fit model
tuning_history = model.fit_generator(
	myImageGenerator.flow(X_train, y_train, batch_size=BATCHSIZE),
	samples_per_epoch=X_train.shape[0],
	epochs=EPOCHS,
	verbose=VERBOSE,
	validation_split=SPLIT
)

