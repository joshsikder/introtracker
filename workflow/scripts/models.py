from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf

def createCNN():
    pass

def createDense(dims, n_class):
    inputs = layers.Input(dims)
    model = layers.Flatten()(inputs)
    model = layers.Dense(256, activation="relu")(model)
    model = layers.Dense(128, activation="relu")(model)
    model = layers.Dense(64, activation="relu")(model)
    model = layers.Dense(32, activation="relu")(model)
    model = layers.Dense(16, activation="relu")(model)
    class_output = layers.Dense(n_class, activation="softmax", name="class_output")(model)

    model = Model(inputs=[inputs], outputs=[class_output], name="DenseTreeSeq")
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
    optimizer="adam",metrics=["accuracy"])
    
    return model

def createRF():
    pass
