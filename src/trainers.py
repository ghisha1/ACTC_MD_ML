# Import packages
import os
import numpy as np
import time
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
#from source.data_preprocessing import DataPreprocessing

class Trainers():
    def __init__(self):
        pass


    def trainSVR_SingleOutput(self, h_params, X_train, X_test, y_test, y_train, folds=5):
        """
        It takes in hyperparameters, training and testing data, and returns the mean and standard
        deviation of the cross validation scores, and the R^2 score of the model.
        
        :param h_params: hyperparameters
        :param X_train: the training data
        :param X_test: the test data
        :param y_test: the actual values of the target variable
        :param y_train: the training data
        :param folds: number of folds for cross validation, defaults to 5 (optional)
        :return: The mean and standard deviation of the cross validation scores, and the r_score.
        """
        print(h_params)

        kernel = 'rbf'
        C = h_params['C']
        #gamma = 'auto'
        epsilon = h_params['epsilon']

        #model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)

        if folds is not None:
            scores = cross_val_score(model, X_train, y_train, cv=folds)

            model.fit(X_train, y_train)

            r_score = model.score(X_test, y_test)

            return scores.mean(), scores.std(), r_score

        else:

            model.fit(X_train, y_train)

            return model

    def trainANN_SingleOutput(self, h_params, X_train, X_test, y_test, y_train, folds=5):
         """
         It takes in hyperparameters, training and testing data, and returns the mean and standard
         deviation of the cross validation scores, and the R^2 score of the model.
         
         :param h_params: hyperparameters
         :param X_train: the training data
         :param X_test: the test data
         :param y_test: the actual values of the target variable
         :param y_train: the training data
         :param folds: number of folds for cross validation, defaults to 5 (optional)
         :return: The mean and standard deviation of the cross validation scores, and the r_score.
        """
         #print(h_params)
         alpha                      = h_params['alpha']
         learning_rate_init         = h_params['learning_rate_init']
         activation_idx             = h_params['activation_idx']
         solver_idx                 = h_params['solver_idx']
         learning_rate_idx          = h_params['learning_rate_idx']
         
         activation                 = ['logistic', 'tanh', 'relu']
         solver                     = ['lbfgs', 'adam']
         learning_rate              = ['constant', 'invscaling', 'adaptive']
         
         
         model = MLPRegressor(alpha=alpha, learning_rate_init=learning_rate_init, learning_rate = learning_rate[learning_rate_idx],
                              activation = activation[activation_idx], solver=solver[solver_idx], max_iter = 5000, n_iter_no_change = 200)

         if folds is not None:
             scores = cross_val_score(model, X_train, y_train, cv=folds)

             model.fit(X_train, y_train)

             r_score = model.score(X_test, y_test)

             return scores.mean(), scores.std(), r_score

         else:

             model.fit(X_train, y_train)

             return model


    def trainRFR_SingleOutput(self, h_params, X_train, X_test, y_test, y_train, folds=5):
         """
         It takes in hyperparameters, training and testing data, and returns the mean and standard
         deviation of the cross validation scores, and the R^2 score of the model.
         
         :param h_params: hyperparameters
         :param X_train: the training data
         :param X_test: the test data
         :param y_test: the actual values of the target variable
         :param y_train: the training data
         :param folds: number of folds for cross validation, defaults to 5 (optional)
         :return: The mean and standard deviation of the cross validation scores, and the r_score.
         """

         max_depth                  = h_params['max_depth']
         n_estimators               = h_params['n_estimators']
         max_features_idx           = h_params['max_features_idx']
         min_weight_fraction_leaf   = h_params['min_weight_fraction_leaf']
         bootstrap_idx              = h_params['bootstrap_idx']
         
         max_features = ['auto','sqrt','log2']
         bootstrap = ['True', 'False']

         model = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators, max_features = max_features[max_features_idx], min_weight_fraction_leaf = min_weight_fraction_leaf,bootstrap = bootstrap[bootstrap_idx])

         if folds is not None:
             scores = cross_val_score(model, X_train, y_train, cv=folds)

             model.fit(X_train, y_train)

             r_score = model.score(X_test, y_test)

             return scores.mean(), scores.std(), r_score

         else:

             model.fit(X_train, y_train)

             return model
