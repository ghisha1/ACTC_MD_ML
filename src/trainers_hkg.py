# Import packages
import os
import numpy as np
import time
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from data_preprocessing import DataPreprocessing

class Trainers(DataPreprocessing):
    def __init__(self):
        pass

    # takes in a module and applies the specified weight initialization

    def weights_init_uniform_rule(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

        if classname.find('Conv1d') != -1:
            # get the number of the inputs
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    # Train

    def trainANN(self, net, optimizer, criterion, trainloader, testloader,
                 epochs):
        print('Training started...')

        train_loss = []
        test_loss = []
        for epoch in range(epochs):

            net.train()
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                x, labels = data

                optimizer.zero_grad()

                if net.__class__.__name__ == 'FCNet' or net.__class__.__name__ == 'FCNetOpt':

                    outputs = net(x.float())  # .to(device)

                else:
                    outputs = net(x.float().unsqueeze(1))  # .to(device)

                labels = labels.float()  # .to(device)

                loss = criterion(outputs, labels)
                # loss.backward(retain_graph=True)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            loss = running_loss / len(trainloader)
            train_loss.append(loss)

            if (epoch % 10) == 0:
                print('Epoch {} of {}, Train Loss: {:.4f}'.format(
                    epoch + 1, epochs, loss))

            net.eval()
            with torch.no_grad():
                running_tloss = 0.0

                for i, data in enumerate(testloader, 0):

                    x, labels = data

                    if net.__class__.__name__ == 'FCNet' or net.__class__.__name__ == 'FCNetOpt':

                        outputs = net(x.float())  # .to(device)

                    else:
                        outputs = net(x.float().unsqueeze(1))  # .to(device)

                    labels = labels.float()  # .to(device)

                    tloss = criterion(outputs, labels)

                    running_tloss += tloss.item()

                tloss = running_tloss / len(testloader)
                test_loss.append(tloss)

                if (epoch % 10) == 0:
                    print('Epoch {} of {}, Test Loss: {:.4f}'.format(
                        epoch + 1, epochs, tloss))

        print('Training completed')
        return train_loss, test_loss

    def trainSVR(self, h_params, X_train, X_test, y_test, y_train, folds=5):
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



    def trainRFR(self, h_params, X_train, X_test, y_test, y_train, folds=5):
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
