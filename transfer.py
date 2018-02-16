from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import backend as K

IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 3 #isn't this very small?
BAT_SIZE = 32 #or whatever
FC_SIZE = 1024
N_CLASSES = 12
FROZEN = 172
SPLIT = 0.2



  # test_datagen = ImageDataGenerator(
  #     preprocessing_function=preprocess_input,
  #     rotation_range=30,
  #     width_shift_range=0.2,
  #     height_shift_range=0.2,
  #     shear_range=0.2,
  #     zoom_range=0.2,
  #     horizontal_flip=True
  # )


# book uses rotation range 40, and fill_mode='nearest', also doesn't use zoom range
#nor do they use a preprocessing function
base_model = InceptionV3(weights="imagenet", include_top=False)

x = base_model.ouput
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(12,, activation="softmax")(x)
model = (input=base_model.input, output=predictions)

for layer in base_model.layers:
	layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# do I need to use np_utils.to_categorical?  Keep that in mind.
# Ah, that final layer is just an aggregator for the N classes.
# validation_split = 0.2 seems fine to me.
# so we have 3 epochs, and 32 batch size...we *don't* have anything for the training/validation split.
# we may not need anything, since we basically odn't care.


  # nb_train_samples = get_nb_files(args.train_dir)
  # nb_classes = len(glob.glob(args.train_dir + "/*"))
  # nb_val_samples = get_nb_files(args.val_dir)
  # nb_epoch = int(args.nb_epoch)
  # batch_size = int(args.batch_size)

#now here in the book, we don't have the training samples thing...so maybe I can steer clear of that.
"""
history = model.fit_generator(
	myImageDataGenerator.flow(X_train, y_train, batch_size=32),
	samples_per_epoch=X_train.shape[0],
	epochs=NB_EPOCH,
	evaluation_split=
	verbose=True
)
"""
# do some training here, need to figure out what

  # history_tl = model.fit_generator(
  #   train_generator,
  #   nb_epoch=nb_epoch,
  #   samples_per_epoch=nb_train_samples,
  #   validation_data=validation_generator,
  #   nb_val_samples=nb_val_samples,
  #   class_weight='auto')



for layer in model.layers[:FROZEN]:
	layer.trainable = False

for layer in model.layers[FROZEN:]:
	layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

  # history_ft = model.fit_generator(
  #   train_generator,
  #   samples_per_epoch=nb_train_samples,
  #   nb_epoch=nb_epoch,
  #   validation_data=validation_generator,
  #   nb_val_samples=nb_val_samples,
  #   class_weight='auto')
