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
import matplotlib.pyplot as plt
from datagenerator import DataGenerator

#2 sets of conv/pooling layers to start, try up to 6 sets
#flatten, then pass to dense layer
#output
#sigmoid for losses

parser = argparse.ArgumentParser(description='Keras model for detecting introgression')
parser.add_argument( '-bs', help = "Batch size for data generator",dest='BATCHSIZE')
parser.add_argument( '-ds', help = "Dataset name",dest='DATASET')
parser.add_argument( '-l', help = "length",dest='LENGTH',type=int)
parser.add_argument( '-out', help = "File Location of saved model(.h5)",dest='OUTPUT')
args = parser.parse_args()

#CONFIG
ntaxa = 3
# loci_count = 100
# loci_length = 1200
# aln_length = loci_count * loci_length
# Parameters
params = {'dim': (ntaxa,args.LENGTH,1),
          'batch_size': int(args.BATCHSIZE),
          'arraypath': f"/work/users/j/s/jsikder/fordan/{args.DATASET}/",
          'labelCorrection': 0,
          'n_classes': 14,
          'n_channels': 1,
          'shuffle': True}

# partition = {'train': [f'{i}_train' for i in range(0,300)], 
#              'validation': [f'{i}_test' for i in range(0,100)]} #tree 1,4
partition = {'train': [f'{i}_train' for i in range(0,29334)],     
             'validation': [f'{i}_test' for i in range(0,14666)]} #tree 5,6
# partition = {'train': ['id-'+str(i) for i in range(0,22001)], 
#              'validation': ['id-'+str(i) for i in range(22001,33001)]} #tree 2,3
             
training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)

seq_inputs = keras.Input(shape=(ntaxa, args.LENGTH, 1))
model = layers.Conv2D(128, 3, strides = 1, padding = "same", activation="relu")(seq_inputs)
model = layers.MaxPooling2D((1,2))(model)
model = layers.Conv2D(128, 3, strides = 1, padding = "same", activation="relu")(model)
model = layers.MaxPooling2D((1,2))(model)
model = layers.Conv2D(128, 3, strides = 1, padding = "same", activation="relu")(model)
model = layers.MaxPooling2D((1,2))(model)
model = layers.Conv2D(128, 3, strides = 1, padding = "same", activation="relu")(model)
model = layers.MaxPooling2D((1,2))(model)
model = layers.Flatten()(model)
model = layers.Dense(14, activation="softmax")(model)

myModel = Model(seq_inputs, model)
print(f'Running on subset {params["arraypath"].split("/")[-2]} with model OriginalCNN')
myModel.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer="adam",metrics=["accuracy"])
myModel.summary()

# earlystop=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
# checkpoint = ModelCheckpoint('best_weights_batch_'+str(args.BATCHSIZE), monitor='val_loss', 
# verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# myModel.fit(x=training_generator, validation_data=validation_generator, callbacks=[checkpoint,earlystop], epochs=50)
myModel.fit(x=training_generator, validation_data=validation_generator, epochs=50)
# history = myModel.fit(x=training_generator, validation_data=validation_generator, epochs=50)
# myModel.save(str(args.OUTPUT) + f"model_bs{args.BATCHSIZE}_introprop{args.SCENARIO}.h5")

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.savefig('plots/accuracy.png')

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.savefig('plots/loss.png')
