
import pandas as pd
import numpy as np
import pickle
import string
import matplotlib.pyplot as plt

# use created functions

class trainedMODELs():
    """

        Contains methods for data acquisition and preprocessing 

    """
    def __init__(self) -> None:
        pass

    def implementation(self, model, type_data, xtrain, xtest):
        """
        # options
        model = 'SVR', 'RFR', 'ANN'
        type_data = 'A0', 'A1', 'A2'
        """

        # load alphabets: alphabets = list(string.ascii_uppercase)[:len(list(y01_MF_train.columns))]

        # descriptors
        desc = {
                    "A": 'gr_minima_Ion_H2O',       "B": 'gr_peak_position_Ion_H2O',
                    "C": 'gr_peak_height_Ion_H2O',  "D": 'Nr_Ion_H2O', 
                    "E": 'gr_minima_CG_H2O',        "F": 'gr_peak_position_CG_H2O', 
                    "G": 'gr_peak_height_CG_H2O',   "H": 'Nr_CG_H2O_',
                    "I": 'gr_minima_CG_Ion',        "J": 'gr_peak_position_CG_Ion', 
                    "K": 'gr_peak_height_CG_Ion',   "L": 'Nr_CG_Ion'
                }

        results_train = pd.DataFrame({}, columns = list(desc.values()))
        results_test = pd.DataFrame({}, columns = list(desc.values()))

        for (tag, ID) in zip(list(desc.keys()), list(desc.values())):

            filename = f'../models/solvation_descriptors/{model}/{type_data}/SolvationDescriptor_{tag}_{model}_{type_data}.sav'
            loaded_model = pickle.load(open(filename, 'rb'))

            # use model
            results_train[ID] = loaded_model.predict(xtrain)  
            results_test[ID] = loaded_model.predict(xtest)  

        
        return results_train, results_test


    def plot_box_MAE(self, ytrain, ytest, ytrain_pred_list, ytest_pred_list, type_data, title):
          
        # plot figures
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))

        axs[0, 0].grid(False); axs[0, 1].grid(False)
        axs[1, 0].grid(False); axs[1, 1].grid(False)
        axs[2, 0].grid(False); axs[2, 1].grid(False)

        ytrain = ytrain.iloc[:, :ytrain.shape[1]]; ytest = ytest.iloc[:, :ytest.shape[1]]
        
        box_00 = axs[0, 0].boxplot((np.abs(ytrain.to_numpy() - ytrain_pred_list[0].to_numpy())), patch_artist = True, notch = False, showfliers = False)
        box_01 = axs[0, 1].boxplot((np.abs(ytest.to_numpy() - ytest_pred_list[0].to_numpy())), patch_artist = True, notch = False, showfliers = False)

        box_10 = axs[1, 0].boxplot((np.abs(ytrain.to_numpy() - ytrain_pred_list[1].to_numpy())), patch_artist = True, notch = False, showfliers = False)
        box_11 = axs[1, 1].boxplot((np.abs(ytest.to_numpy() - ytest_pred_list[1].to_numpy())), patch_artist = True, notch = False, showfliers = False)

        box_20 = axs[2, 0].boxplot((np.abs(ytrain.to_numpy() - ytrain_pred_list[2].to_numpy())), patch_artist = True, notch = False, showfliers = False)
        box_21 = axs[2, 1].boxplot((np.abs(ytest.to_numpy() - ytest_pred_list[2].to_numpy())), patch_artist = True, notch = False, showfliers = False)

        # set colors
        colors = [
                    '#0000FF', '#00FF00', '#FFFF00', '#DE3163', 
                    '#00FFFF', '#00FFF0', '#6495ED', '#FFF0FF', 
                    '#000FFF', '#0FFF00', '#FFFF0F', '#FF7F50'
                ]
        for patch, color in zip(box_00['boxes'], colors): patch.set_facecolor(color)
        for patch, color in zip(box_01['boxes'], colors): patch.set_facecolor(color)
        for patch, color in zip(box_10['boxes'], colors): patch.set_facecolor(color)
        for patch, color in zip(box_11['boxes'], colors): patch.set_facecolor(color)
        for patch, color in zip(box_20['boxes'], colors): patch.set_facecolor(color)
        for patch, color in zip(box_21['boxes'], colors): patch.set_facecolor(color)

        # update the text formats
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 10

        # set labels
        fig.supylabel(r'$\rm MAE $', fontsize = 22, fontweight = 'bold')
        fig.supxlabel(r'$\rm Solvation \ descriptors $', fontsize = 22, fontweight = 'bold')

        axs[0, 0].set_title(r'$\rm Train$', fontsize = 22, fontweight = 'bold')
        axs[0, 1].set_title(r'$\rm Test$', fontsize = 22, fontweight = 'bold')

        alphabets = list(string.ascii_uppercase)[:len(list(ytrain.columns))]
        axs[2, 0].set_xticklabels(labels=[f'{i}' for i in alphabets], fontsize='x-large') 
        axs[2, 1].set_xticklabels(labels=[f'{i}' for i in alphabets], fontsize='x-large') 

        axs[0, 0].set_ylabel(r'$SVR$', labelpad = 5, fontsize = 15, fontweight='bold')
        axs[1, 0].set_ylabel(r'$ANN$', labelpad = 5, fontsize = 15, fontweight='bold')
        axs[2, 0].set_ylabel(r'$RFR$', labelpad = 5, fontsize = 15, fontweight='bold')

        # set axis
        axs[0, 0].set_xticks([]); axs[1, 0].set_xticks([])
        axs[0, 1].set_xticks([]); axs[1, 1].set_xticks([])

        if type_data == 'A0':
            axs[0, 0].set_ylim([-0.02, 0.3]); axs[1, 0].set_ylim([-0.02, 0.3]); axs[2, 0].set_ylim([-0.02, 0.3])
            axs[0, 1].set_ylim([-0.02, 0.7]); axs[1, 1].set_ylim([-0.02, 0.7]); axs[2, 1].set_ylim([-0.02, 0.7])

        if type_data == 'A1':
            axs[0, 0].set_ylim([-0.02, 0.5]); axs[1, 0].set_ylim([-0.02, 0.5]); axs[2, 0].set_ylim([-0.02, 0.5])
            axs[0, 1].set_ylim([-0.02, 1.0]); axs[1, 1].set_ylim([-0.02, 1.0]); axs[2, 1].set_ylim([-0.02, 1.0])
        
        if type_data == 'A2':
            axs[0, 0].set_ylim([-0.02, 0.6]); axs[1, 0].set_ylim([-0.02, 0.6]); axs[2, 0].set_ylim([-0.02, 0.6])
            axs[0, 1].set_ylim([-0.02, 1.5]); axs[1, 1].set_ylim([-0.02, 1.5]); axs[2, 1].set_ylim([-0.02, 1.5])
    

        plt.tight_layout()
        plt.show
        #plt.savefig(f'figures/{title}.png', dpi=600) 







