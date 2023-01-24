import sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow import keras
import sklearn.model_selection as modelselect
import matplotlib.pyplot as plt
import pydot
import graphviz
import pydotplus
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import models

#man graphviz is stupid, add everytime
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

def cnn_cd():
    train_dir = r'C:\Personal_Data\cnn_udemy\catsdogs\train'
    test_dir = r'C:\Personal_Data\cnn_udemy\catsdogs\test'
    val_dir = r'C:\Personal_Data\cnn_udemy\catsdogs\validation'

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=40, class_mode= 'binary')
    val_gen = test_datagen.flow_from_directory(val_dir, target_size=(150,150), batch_size=40, class_mode= 'binary')

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
                  metrics=["accuracy"])

    fitter = model.fit_generator(
        train_gen,
        steps_per_epoch=50,
        epochs=30,
        validation_data=val_gen,
        validation_steps=25)

    pd.DataFrame(fitter.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    model.save("model_cat_dog_cnn_aug.h5")


if __name__ == '__main__':
    cnn_cd()