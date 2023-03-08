# import library
import sys
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import OneHotEncoder
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import (get_crossover, get_mutation, get_sampling,
                           get_termination)
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
import seaborn as sns
from src.trainers import Trainers

class svrModel(ElementwiseProblem):
    def __init__(self, X_train, X_test, y_train, y_test, **kwargs):
        vars = {
            "C": Real(bounds = (0.1, 100)),
            "epsilon": Real(bounds = (0.01, 20.0))
        }

        super().__init__(vars=vars, n_obj=3, n_ieq_constr=0, **kwargs)

        self.trainer = Trainers()

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def _evaluate(self, X, out, *arg, **kwargs):

        scores = self.trainer.trainSVR_SingleOutput(h_params= X,
                                       X_train=self.X_train,
                                       X_test=self.X_test,
                                       y_train=self.y_train,
                                       y_test=self.y_test,
                                       folds=5)

        out['F'] = [-1 * scores[0], scores[1], -1 * scores[2]]


class svrSolvers(ElementwiseProblem):
    def __init__(self):
        super(ElementwiseProblem, self).__init__()

    def svrSolver(self, X_train, X_test, y_train, y_test):

        problem = svrModel(X_train, X_test, y_train, y_test)

        algorithm = NSGA2(  
                        pop_size=2,
                        sampling=MixedVariableSampling(),
                        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                        eliminate_duplicates=MixedVariableDuplicateElimination(),
                        n_offsprings=4
                        )

        res = minimize(problem,
                       algorithm,
                       termination=("n_gen", 500),
                       seed=1,
                       save_history=True,
                       verbose=True)

        return res

    def get_best(self, res):

        for i in range(len(res.F)):

            w_metric = -1 * res.F[:, 0] - res.F[:, 1] - res.F[:, 2]

            if w_metric[i] == max(w_metric[:]):
                best_x = res.X[i]
                best_f = res.F[i]

        return best_x, best_f

class annModel(ElementwiseProblem):
    def __init__(self, X_train, X_test, y_train, y_test, **kwargs):
        vars = {
            "alpha": Choice(options = [0.0001, 0.001, 0.01, 0.1, 1]),
            "learning_rate_init": Choice(options = [0.001, 0.01, 0.1, 1.0]),
            "activation_idx":Integer(bounds = (0, 2)),
            'solver_idx': Integer(bounds = (0, 1)),
            "learning_rate_idx": Integer(bounds=(0, 2))
        }


        super().__init__(vars=vars, n_obj=3, n_ieq_constr=0, **kwargs)

        self.trainer = Trainers()
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def _evaluate(self, X, out, *arg, **kwargs):

        scores = self.trainer.trainANN_SingleOutput(h_params= X,
                                       X_train=self.X_train,
                                       X_test=self.X_test,
                                       y_train=self.y_train,
                                       y_test=self.y_test,
                                       folds=5)

        out['F'] = [-1 * scores[0], scores[1], -1 * scores[2]]


class annSolvers(ElementwiseProblem):
    def __init__(self):
        super(ElementwiseProblem, self).__init__()

    def annSolver(self, X_train, X_test, y_train, y_test):

        problem = annModel(X_train, X_test, y_train, y_test)

        algorithm = NSGA2(  
                        pop_size=2,
                        sampling=MixedVariableSampling(),
                        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                        eliminate_duplicates=MixedVariableDuplicateElimination(),
                        n_offsprings=4
                        )


        res = minimize(problem,
                       algorithm,
                       termination=("n_gen",10),
                       seed=1,
                       save_history=True,
                       verbose=True)

        return res

    def get_best(self, res):

        for i in range(len(res.F)):

            w_metric = -1 * res.F[:, 0] - res.F[:, 1] - res.F[:, 2]

            if w_metric[i] == max(w_metric[:]):
                best_x = res.X[i]
                best_f = res.F[i]

        return best_x, best_f

class rfrModel(ElementwiseProblem):
    def __init__(self, X_train, X_test, y_train, y_test, **kwargs):
        vars = {
            'max_depth': Integer(bounds = (1, 100)),
            'n_estimators':Integer(bounds = (1, 100)),
            'max_features_idx': Integer(bounds = (0, 2)),
            'min_weight_fraction_leaf':Real(bounds = (0, 1)),
            'bootstrap_idx':Integer(bounds = (0,1))
            
            
        }

        super().__init__(vars=vars, n_obj=3, n_ieq_constr=0, **kwargs)

        self.trainer = Trainers()
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def _evaluate(self, X, out, *arg, **kwargs):

        scores = self.trainer.trainRFR_SingleOutput(h_params= X,
                                       X_train=self.X_train,
                                       X_test=self.X_test,
                                       y_train=self.y_train,
                                       y_test=self.y_test,
                                       folds=5)

        out['F'] = [-1 * scores[0], scores[1], -1 * scores[2]]


class rfrSolvers(ElementwiseProblem):
    def __init__(self):
        super(ElementwiseProblem, self).__init__()

    def rfrSolver(self, X_train, X_test, y_train, y_test):

        problem = rfrModel(X_train, X_test, y_train, y_test)

        #change the pop-size, n_offsprings, n_gen (if needed)
        algorithm = NSGA2(  
                        pop_size=2,
                        sampling=MixedVariableSampling(),
                        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                        eliminate_duplicates=MixedVariableDuplicateElimination(),
                        n_offsprings=2
                        )


        res = minimize(problem,
                       algorithm,
                       termination=("n_gen", 200),
                       seed=1,
                       save_history=True,
                       verbose=True)

        return res

    def get_best(self, res):

        for i in range(len(res.F)):

            w_metric = -1 * res.F[:, 0] - res.F[:, 1] - res.F[:, 2]

            if w_metric[i] == max(w_metric[:]):
                best_x = res.X[i]
                best_f = res.F[i]

        return best_x, best_f

