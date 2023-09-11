#!/usr/bin/env python
import sys, os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import models
from genutils import GeneralUtilities as gu

def createMonitors(data, ds):    
    earlystop=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(f'{data["weights_output"]}/SPF_best_weights_{ds}.wt', monitor='val_loss',
        verbose=0, save_best_only=True, save_weights_only=False, mode='auto', 
        period=1)

def runModel(model, X, y, data, ds, es = False, cp = False):
    if es == True:
        earlystop=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    if cp == True:
        checkpoint = ModelCheckpoint(f'{data["weights_output"]}/{sys.argv[0].rstrip(".py")}_best_weights_{ds}', 
            monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
            mode='auto')
    if es == True and cp == True:
        model.fit(x=X, y=y, validation_split=0.2, 
                    callbacks=[checkpoint,earlystop], 
                    epochs=data['epochs'])
    else:
        model.fit(x=X, y=y, validation_split=0.2, epochs=data['epochs'])

def main():
    parser = argparse.ArgumentParser(description='Keras model for detecting introgression')
    parser.add_argument( '-cf', '--configfile', help = "Configuration file",dest='CFGFILE')
    parser.add_argument( '-c', '--config', help = "Dataset name",dest='CFG')
    parser.add_argument( '--run', '-r', help = "Run model (absence will dryrun the model)",dest='runModel', action='store_true')
    parser.add_argument( '--save', help = "Save model",dest='saveModel', action='store_true')
    parser.add_argument( '--summary', help = "Display model summary",dest='modelSummary', action='store_true')
    args = parser.parse_args()

    dataset = gu.readMultiConfig(args.CFGFILE, args.CFG)

    arr=np.load(f'{dataset["inputs"]}')

    X = arr["X"]
    y = arr["y"]

    indices = np.random.permutation(int(y.shape[0]))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    createMonitors(dataset, args.CFG)
    myModel = models.createDense((X_shuffled.shape[1],), dataset['n_classes'])
    
    if args.modelSummary == True:
        myModel.summary()
    if args.runModel == True:
        runModel(myModel, X_shuffled, y_shuffled, dataset, args.CFG, es = False, cp = False)
    if args.saveModel == True:
        myModel.save(f"{dataset['model_output']}/{sys.argv[0].rstrip('.py')}_introprop{args.SCENARIO}.h5")

if __name__ == '__main__':
    main()