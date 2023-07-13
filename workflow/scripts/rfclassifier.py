from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import argparse
import numpy as np

def loadData(input):
    arr = np.load(input)
    return(arr['X'], arr['y'])

def main():
    parser = argparse.ArgumentParser(description='Splits data to be passed as input to the data generator for Keras')
    parser.add_argument( '-i', '--indata', help = "Input data",dest='DATA')
    args = parser.parse_args()

    X, y = loadData(args.DATA)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    param_dist = {'n_estimators': randint(50,500),
                  'max_depth': randint(1,200),
                  'min_samples_split': [2,5,10,15,20,30],
                  'min_samples_leaf': [1,2,3,4],
                  'bootstrap': [True, False],
                  'max_features': [None, 'sqrt', 'log2'],
                  'criterion': ['gini','entropy','log_loss']}

    rf = RandomForestClassifier(n_jobs=16)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions = param_dist, \
    n_iter = 30, cv = 5, n_jobs = 16)

    rf.fit(X_train, y_train)
    rf_random.fit(X_train, y_train)

    best_rf = rf_random.best_estimator_

    y_pred = rf.predict(X_test)
    y_pred_best = rf_random.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    accuracy_best = accuracy_score(y_test, y_pred_best)
    precision_best = precision_score(y_test, y_pred_best, average='macro')
    recall_best = recall_score(y_test, y_pred_best, average='macro')
    print(f"{args.DATA} Base Accuracy: {accuracy}, Base Precision: {precision}, Base Recall: {recall}")
    print(f"{args.DATA} Best Accuracy: {accuracy_best}, Best Precision: {precision_best}, Best Recall: {recall_best}")
    print(f"Best hyperparameters: {rf_random.best_params_}\n")

if __name__ == '__main__':
    main()