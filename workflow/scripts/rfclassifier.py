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

    rf = RandomForestClassifier(n_jobs=16)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    print(f"{args.DATA} Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

if __name__ == '__main__':
    main()