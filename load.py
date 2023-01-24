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

def load_model():
    model = keras.models.load_model("ANN_Image_Class.h5")

    fashion_mnist = keras.datasets.fashion_mnist
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    classes = ["T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
               "Sneaker", "Bag", "Ankle Boot"]
    X_train_norm = X_train / 255.
    X_test_norm = X_test / 255.

    tr_val_X, ts_val_X, tr_val_Y, ts_val_Y = modelselect.train_test_split(X_train_norm,
                                                                          Y_train, test_size=0.15, random_state=24061)

    fitter = model.fit(tr_val_X, tr_val_Y, epochs=30, validation_data=(ts_val_X, ts_val_Y))

    # fitter.params  fitter.history

    pd.DataFrame(fitter.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    scr = model.evaluate(X_test_norm, Y_test)
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    class_predict = np.array(classes)[y_pred]

    print(model.summary)
    jj = 56

if __name__ == '__main__':
    load_model()