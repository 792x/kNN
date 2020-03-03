#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from utils import time_diff

def load_data():
    print(f'\n{"=" * 30}\nLoading data...\n{"=" * 30}')
    start = datetime.utcnow()

    # Load the csv files into train and test dataframes
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), "MNIST_train_small.csv")).values
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), "MNIST_test_small.csv")).values

    # Reshape and normalize (between 0 and 1) training data
    trainX = train[:, 1:].reshape(train.shape[0], 1, 28, 28).astype('float32')
    X_train = trainX / 255.0

    y_train = train[:, 0]

    # Reshape and normalize (between 0 and 1) test data
    testX = test[:, 1:].reshape(test.shape[0], 1, 28, 28).astype('float32')
    X_test = testX / 255.0

    y_test = test[:, 0]

    print(f'Loading finished in {time_diff(start)} ms.\n')
    return X_train, y_train, X_test, y_test

def train():
    pass


def main():
    """ Main entry point of the app """
    print("Starting application...")
    X_train, y_train, X_test, y_test = load_data()


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
