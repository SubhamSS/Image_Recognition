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

#man graphviz is stupid, add everytime
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

def img():
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train,Y_train),(X_test,Y_test) = fashion_mnist.load_data()
    classes = ["T-Shirt/Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt",
               "Sneaker","Bag","Ankle Boot"]
    X_train_norm = X_train/255.
    X_test_norm = X_test/255.

    tr_val_X, ts_val_X, tr_val_Y, ts_val_Y = modelselect.train_test_split(X_train_norm,
                                                                          Y_train, test_size=0.15, random_state=24061)
    np.random.seed(42)
    tf.random.set_seed(42)
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28,28])) #convert to 1 array of 784 elements
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    keras.utils.plot_model(model)

    #weights, biases = model.layers[1].get_weights()

    #sparse as labels
    model.compile(loss='sparse_categorical_crossentropy',optimizer="sgd",metrics=["accuracy"])

    #checkpoint_cb = keras.callbacks.ModelCheckpoint("Model - {epoch:02d}.h5")

    checkpoint_cb = keras.callbacks.ModelCheckpoint("Model_Best_early.h5",save_best_only=True)

    early_stop_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

    fitter = model.fit(tr_val_X,tr_val_Y, epochs=200,validation_data=(ts_val_X,ts_val_Y), callbacks=[checkpoint_cb, early_stop_cb])

    #fitter.params  fitter.history

    pd.DataFrame(fitter.history).plot(figsize = (8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

    scr = model.evaluate(X_test_norm,Y_test)
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    class_predict = np.array(classes)[y_pred]

    model.save('ANN_Image_Class.h5')


if __name__ == '__main__':
    img()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
