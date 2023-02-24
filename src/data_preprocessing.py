import pickle
from pickle import dump, load
import time
import warnings
import os
from os import path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from torch.utils.data import DataLoader
from urllib.request import urlopen
from urllib.parse import quote

#from data_load import data_load
from plotters import Plotters
#from plotters import Plotters
#from device_models.edi_model_npj_v2_1 import EDI_Cell

#requred packages for MDFP+
import numpy
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)
start_time = time.time()


class DataPreprocessing(Plotters):
    """

        Contains methods for data acquisition and preprocessing 

        Inherits from plotters to produce figures

        **EEM is an external resource

    """
    def __init__(self):
        pass

    def data_load(self, filename: str, sheetname=0, columns=None):
        """
        It takes a file name, a sheet name, and a list of columns as arguments, and returns a dataframe
        
        :param filename: str = name of the file
        :type filename: str
        :param sheetname: The name of the sheet in the excel file, defaults to 0 (optional)
        :param columns: list of columns to be used in the model
        :return: A dataframe
        """

        # Set source as working directory
        try:
            os.chdir('src/models')
        except:
            if 'models' in os.getcwd():
                os.chdir('../../src/models')
            else:
                os.chdir('../src/models')

        os.chdir('../..')
        d = os.getcwd()
        o = [
            os.path.join(d, o) for o in os.listdir(d)
            if os.path.isdir(os.path.join(d, o))
        ]
        for item in o:
            # print(o)
            if os.path.exists(item + '\\processed\\' + filename):
                file = item + '\\processed\\' + filename

        if file.endswith('.xlsx'):
            data = pd.read_excel(file, sheet_name=sheetname, usecols=columns)

        elif file.endswith('.json'):
            cols = []

            if columns is None:
                data = pd.read_json(file, lines=True)

            else:
                for i in columns:
                    cols.append(i)

                data = pd.read_json(file).filter(cols)

        os.chdir('./src/models')

        return data


#-----------------------------------------------------------------###
