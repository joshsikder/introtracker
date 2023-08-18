#!/usr/bin/env python
import sys
import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from datagenerator import DataGenerator

parser = argparse.ArgumentParser(description='Keras model for detecting introgression')
parser.add_argument( '-a', '--array', help = "Batch size for data generator",dest='INPUT')
parser.add_argument( '-s', '--scenario', help = "Introgression scenario",dest='SCENARIO')
parser.add_argument( '-o', '--output', help = "File Location of saved model(.h5)",dest='OUTPUT')
args = parser.parse_args()

arr=np.load(f'{args.INPUT}')
X = arr["X"]
y = arr["y"]

indices = np.random.permutation(int(y.shape[0]))
X_shuffled = X[indices]
y_shuffled = y[indices]

seq_inputs = keras.Input(shape=(X_shuffled.shape[1],))
# model = layers.Flatten()(seq_inputs)
model = layers.Dense(256, activation="relu")(seq_inputs)
model = layers.Dense(128, activation="relu")(model)
model = layers.Dense(64, activation="relu")(model)
model = layers.Dense(32, activation="relu")(model)
model = layers.Dense(16, activation="relu")(model)
model = layers.Dense(4, activation="softmax")(model)
myModel = Model(seq_inputs, model)

myModel.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer="adam",metrics=["accuracy"])
myModel.summary()

earlystop=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(f'{args.OUTPUT}/best_weights_{args.SCENARIO}.wt', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

myModel.fit(x=X_shuffled, y=y_shuffled, validation_split=0.2, callbacks=[checkpoint,earlystop], epochs=50)
myModel.save(f"{args.OUTPUT}/model_introprop{args.SCENARIO}.h5")
