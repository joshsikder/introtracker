#!/usr/bin/env python
import sys, os
import numpy as np
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from datagenerator import DataGenerator

parser = argparse.ArgumentParser(description='Keras model for detecting introgression')
parser.add_argument( '-in', help = "Input directory of data",dest='INPUT')
parser.add_argument( '-lc', help = "Label correction",dest='LABELCORRECTION', type=int)
parser.add_argument( '-bs', help = "Batch size for data generator",dest='BATCHSIZE', type=int)
parser.add_argument( '-s', help = "Introgression scenario",dest='SCENARIO')
parser.add_argument( '-out', help = "File Location of saved model(.h5)",dest='OUTPUT')
args = parser.parse_args()

# tensorboard_callback = TensorBoard(log_dir="/nas/longleaf/home/jsikder/introtracker/ignored/logs/")

#CONFIG
ntaxa = 3
loci_count = 100
loci_length = 1200
aln_length = loci_count * loci_length
# Parameters
params = {'dim': (ntaxa,aln_length,1),
          'batch_size': args.BATCHSIZE,
          'arraypath': args.INPUT,
          'labelCorrection': args.LABELCORRECTION,
          'n_classes': 4,
          'n_channels': 1,
          'shuffle': True}

# partition14 = {'train': ['id-'+str(i) for i in range(0,29335)], 
#              'validation': ['id-'+str(i) for i in range(29335,44001)]} #tree 1,4
# partition23 = {'train': ['id-'+str(i) for i in range(0,22001)], 
#              'validation': ['id-'+str(i) for i in range(22001,33001)]} #tree 2,3
partitionTest = {'train': [f'{i}_train' for i in range(0,300)], 
                 'validation': [f'{i}_test' for i in range(0,300)]} #subset data
             
training_generator = DataGenerator(partitionTest['train'], **params)
validation_generator = DataGenerator(partitionTest['validation'], **params)

seq_inputs = keras.Input(shape=(ntaxa, aln_length,1))
model = layers.Flatten()(seq_inputs)
model = layers.Dense(256, activation="relu")(model)
model = layers.Dense(128, activation="relu")(model)
model = layers.Dense(64, activation="relu")(model)
model = layers.Dense(32, activation="relu")(model)
model = layers.Dense(16, activation="relu")(model)
model = layers.Dense(4, activation="softmax")(model)

myModel = Model(seq_inputs, model)

myModel.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer="adam",metrics=["accuracy"])
myModel.summary()

earlystop=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(f'best_weights_batch_{args.BATCHSIZE}.txt', monitor='val_loss', 
verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

myModel.fit(x=training_generator, validation_data=validation_generator, callbacks=[checkpoint,earlystop], epochs=1)
# myModel.save(f"{args.OUTPUT}/model_bs{args.BATCHSIZE}_introprop{args.SCENARIO}.h5")