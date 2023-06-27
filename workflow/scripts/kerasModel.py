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

#2 sets of conv/pooling layers to start, try up to 6 sets
#flatten, then pass to dense layer
#output
#sigmoid for losses

parser = argparse.ArgumentParser(description='Keras model for detecting introgression')
parser.add_argument( '-bs', help = "Batch size for data generator",dest='BATCHSIZE')
parser.add_argument( '-intro', help = "Introgression scenario",dest='SCENARIO')
parser.add_argument( '-out', help = "File Location of saved model(.h5)",dest='OUTPUT')
args = parser.parse_args()

print(tf.test.is_built_with_cuda());print(tf.config.list_physical_devices('GPU'))
#CONFIG
ntaxa = 4
loci_count = 100
loci_length = 1200
aln_length = loci_count * loci_length
# Parameters
params = {'dim': (ntaxa,aln_length,1),
          'batch_size': int(args.BATCHSIZE),
          'n_classes': 14,
          'n_channels': 1,
          'shuffle': True}

partition = {'train': ['id-'+str(i) for i in range(0,29335)], 
             'validation': ['id-'+str(i) for i in range(29335,44001)]} #tree 1,4
# partition = {'train': ['id-'+str(i) for i in range(0,22001)], 
#              'validation': ['id-'+str(i) for i in range(22001,33001)]} #tree 2,3
             
training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)

seq_inputs = keras.Input(shape=(ntaxa, aln_length,1))
model = layers.Conv2D(128, 3, strides = 1, padding = "same", activation="relu")(seq_inputs)
model = layers.MaxPooling2D((1,2))(model)
model = layers.Conv2D(128, 3, strides = 1, padding = "same", activation="relu")(model)
model = layers.MaxPooling2D((1,2))(model)
model = layers.Conv2D(128, 3, strides = 1, padding = "same", activation="relu")(model)
model = layers.MaxPooling2D((1,2))(model)
model = layers.Conv2D(128, 3, strides = 1, padding = "same", activation="relu")(model)
model = layers.MaxPooling2D((1,2))(model)
model = layers.Flatten()(model)
# model = layers.Dense(1000, activation="relu")(model)
model = layers.Dense(14, activation="softmax")(model)

myModel = Model(seq_inputs, model)

myModel.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer="adam",metrics=["accuracy"])
myModel.summary()

earlystop=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
checkpoint = ModelCheckpoint('best_weights_batch_'+str(args.BATCHSIZE), monitor='val_loss', 
verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

myModel.fit(x=training_generator, validation_data=validation_generator, callbacks=[checkpoint,earlystop], epochs=50)
myModel.save(str(args.OUTPUT) + f"model_bs{args.BATCHSIZE}_introprop{args.SCENARIO}.h5")

