#!/usr/bin/env python
import sys, os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datagenerator import DataGenerator
import models
from genutils import GeneralUtilities as gu


def parseConfig(cfgFile, cfg):
    configurations = gu.readKerasConfig(cfgFile)
    return configurations[cfg]

def buildGenerators(config):
    inputs = config['inputs']
    ntaxa = config['ntaxa']
    loci_count = config['loci_count']
    loci_length = config['loci_length']
    n_classes = config['n_classes']
    n_channels = config['n_channels']
    shuffle = config['shuffle']
    labelCorrection = config['labelCorrection']
    batch_size = config['batch_size']
    epochs = config['epochs']
    train_start = config['train_start']
    train_end = config['train_end']
    val_start = config['val_start']
    val_end = config['val_end']

    aln_length = loci_count * loci_length
    params = {'dim': (ntaxa,aln_length,n_channels),
            'batch_size': batch_size,
            'arraypath': inputs,
            'labelCorrection': labelCorrection,
            'n_classes': n_classes,
            'n_channels': n_channels,
            'shuffle': shuffle}

    partitionTest = {'train': [f'{i}_train' for i in range(train_start,train_end)], 
                     'validation': [f'{i}_test' for i in range(val_start,val_end)]}
                
    training_generator = DataGenerator(partitionTest['train'], **params)
    validation_generator = DataGenerator(partitionTest['validation'], **params)
    return training_generator, validation_generator

def runModel(model, trainer, validator, data, es = False, cp = False):
    if es == True:
        earlystop=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    if cp == True:
        checkpoint = ModelCheckpoint(f'best_weights_batch_{data["batch_size"]}.txt', monitor='val_loss', 
            verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    if es == True and cp == True:
        model.fit(x=trainer, validation_data=validator, 
                    callbacks=[checkpoint,earlystop], 
                    epochs=data['epochs'])
    else:
        model.fit(x=trainer, validation_data=validator, 
                  epochs=data['epochs'])

def main():
    parser = argparse.ArgumentParser(description='Keras model for detecting introgression')
    parser.add_argument( '--configfile', '-cf', help = "Configuration file",dest='cfgFile')
    parser.add_argument( '--config', '-c', help = "Settings set",dest='cfg')
    parser.add_argument( '--run', '-r', help = "Run model (absence will dryrun the model)",dest='runModel', action='store_true')
    parser.add_argument( '--save', help = "Save model",dest='saveModel', action='store_true')
    parser.add_argument( '--summary', help = "Display model summary",dest='modelSummary', action='store_true')
    args = parser.parse_args()
    
    dataset = parseConfig(args.cfgFile, args.cfg)
    tgen, vgen = buildGenerators(dataset)
    
    myModel = models.createDense((dataset['ntaxa'],
                                  dataset['loci_count']*dataset['loci_length'],
                                  dataset['n_channels']),
                                  dataset['n_classes'])
    
    if args.modelSummary == True:
        myModel.summary()
    if args.runModel == True:
        runModel(myModel, tgen, vgen, dataset, False, False)
    if args.saveModel == True:
        myModel.save(f"{dataset['output']}/model_bs{dataset['batch_size']}_introprop{args.cfg}.keras")

if __name__ == '__main__':
    main()