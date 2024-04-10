import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

cwd = os.getcwd()
train_dir = os.path.join(cwd, "datasets", "train")
test_dir = os.path.join(cwd, "datasets", "test")
val_dir = os.path.join(cwd, "datasets", "validation")

BATCH_SIZE = 16
IMG_SIZE = 128
EPOCHS = 10

# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     horizontal_flip=True,
#     rotation_range=10,
#     shear_range=0.1,
# )
# val_datagen = ImageDataGenerator(rescale=1.0 / 255)
# test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    shear_range=0.1,
    preprocessing_function=preprocess_input
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=17,
    class_mode="sparse",
)

val_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=17,
    class_mode="sparse",
)
test_generator = test_datagen.flow_from_directory(
    target_size=(IMG_SIZE, IMG_SIZE),
    directory=test_dir, class_mode="sparse"
)
label = ["paper", "rock", "scissor"]

# print(train_generator.filepaths[0])
# print(train_generator.labels[0])
# print(train_generator.next())
Mob = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False)
Mob.trainable = False
# Input = layers.Input(shape=(32, 32, 3))

conv_11 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(Mob.output)
conv_12 = layers.Conv2D(64, (1, 1), padding="same", activation="relu")(conv_11)
maxpool_1 = layers.MaxPooling2D(2, 2)(conv_12)
# conv_21 = layers.Conv2D(128, (3, 3), activation="relu")(maxpool_1)
# conv_22 = layers.Conv2D(128, (1, 1), activation="relu")(conv_21)
# maxpool_2 = layers.MaxPooling2D(2, 2)(conv_22)
# conv_31 = layers.Conv2D(128, (3, 3), activation="relu")(maxpool_1)
# conv_32 = layers.Conv2D(128, (1, 1), activation="relu")(conv_31)
# maxpool_3 = layers.MaxPooling2D(2, 2)(conv_32)
flatten = layers.Flatten()(maxpool_1)
Output = layers.Dense(3, activation="softmax")(flatten)

model = models.Model(inputs=Mob.input, outputs=Output)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(
    train_generator, epochs=EPOCHS, validation_data=val_generator, batch_size=BATCH_SIZE
)
ev = model.evaluate(test_generator, batch_size=BATCH_SIZE)
model.save("model/out_model.model")

