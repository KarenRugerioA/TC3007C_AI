# Importing the necessary libraries for the data analysis and transformations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from joblib import dump

# Function that trains the model mlp
def train_mlp(target_column_name, original_name_dataset):

    # Obtaining the train and test dataset
    train = pd.read_csv(f'../data/{original_name_dataset}/original_train.csv')
    x_test = pd.read_csv(f'../data/{original_name_dataset}/test/x_test.csv')
    y_test = pd.read_csv(f'../data/{original_name_dataset}/test/y_test.csv')

    # Dividing the train dataset
    x_train = train.drop([target_column_name], axis=1)
    y_train = pd.DataFrame(train[target_column_name])

    #Initializing the MLPClassifier hyperparameters
    N_EPOCHS = 100
    classifier = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=N_EPOCHS,activation = 'logistic',solver='adam',random_state=1)

    #Fitting the training data to the network
    classifier.fit(x_train, y_train)

    #Predicting y for X_val
    y_pred_prob = classifier.predict_proba(x_test)
    y_pred = classifier.predict(x_test)

    #Getting the accuracy of the model
    # print(classifier.score(x_test, y_test))

    #Creating a confusion matrix to help determinate accuracy wtih classification model
    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements

    #Evaluataion of the predictions against the actual observations in y_val
    cm = confusion_matrix(y_pred, y_test)

    #Printing the accuracy
    acc = round(accuracy(cm),2)
    percentage = "{:.0%}".format(acc)
    print(f"Accuracy of Model: {percentage}")

    # Confussion Matrix
    print(pd.DataFrame(cm))