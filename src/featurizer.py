# import library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import shap
from sklearn.decomposition import PCA
import seaborn as sns

# Load data, encode the ion
def featurization_FW2(data01_filename, polymers_names, polymers_smiles, descriptorType, approach):

    #****************************************** Load data and encode the ion ************************************************#
    # import training data
    data01=pd.read_excel(data01_filename)
    #data01 = pd.read_csv(data01_filename, encoding='utf-8')
    # convert the categorical variable to numerical values via OneHotEncoder
    ions01 = data01[['CounterIon','Co-Ion']].to_numpy()
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(ions01)
    ohe_ions01 = enc.transform(ions01).toarray()
    ohe_df01 = pd.DataFrame(data = ohe_ions01, columns=[f'ohe_{i}' for i in range(len(ohe_ions01[0]))])
    #
    #****************************************** Generate FingerPrint ************************************************#
    polymers_dict_ = dict(zip(polymers_names, polymers_smiles))
    #
    # Morgan Fingerprint
    if descriptorType == 'MorganDescr':
        moles_dict_ = {}
        for polymer in polymers_names:
            moles_dict_[polymer] = Chem.MolFromSmiles(polymers_dict_[polymer])
        fps_dict = {}
        # generate morgan fingerprints
        for polymer in polymers_names:
            fps_dict[polymer] = AllChem.GetMorganFingerprintAsBitVect(moles_dict_[polymer], useChirality=True, radius=3, nBits=128)
        #
        # convert descriptor dictionary into arrays
        vects_dict = {}
        for polymer in polymers_names:
            data_ = np.array(fps_dict[polymer]).reshape(1, -1)
            vects_dict[polymer] = pd.DataFrame(data = data_, columns=[f'mfp_{i}' for i in range(len(data_[0]))])


    #****************************************** Generate vector ************************************************
    #
    #******************** Method 1 ********************
    #
    # For data01
    fingerprints01 = pd.DataFrame()
    #
    for i, polymer in enumerate(data01['Name of the polymer']):
        poly=polymer.replace('\u200b','')
        try:
            fingerprints01 = pd.concat([vects_dict[poly], fingerprints01.reset_index(drop=True)], axis=0).reset_index(drop=True)
        except KeyError:
            pass
    #
    reverse_fingerprints = fingerprints01.iloc[::-1].reset_index(drop=True)
    
    if approach == 'METHODX':
        # drop unneeded columns # 'Exp_act_coeff'
        data01_m = data01.drop(columns = ['#', 'Name of the polymer', 'CounterIon', 'Co-Ion', 'salt', 'Hopping_rate_Ion',
                                                 'Diffcoeff (Counterion), A^2/ps', 'Diffcoeff (Oxy in H2O), A^2/ps', 'Exp_act_coeff'])
        
        ## merge data - data01 and machine learning model 1
        data01_mm = pd.concat([ohe_df01, data01_m], axis=1).reset_index(drop=True)
        data01_mmm = pd.concat([reverse_fingerprints, data01_mm], axis=1).dropna()

    if approach == 'METHODX_A0':
        # drop unneeded columns # 'Exp_act_coeff'
        data01_m = data01.drop(columns = ['#', 'Name of the polymer', 'CounterIon', 'Co-Ion', 'salt'])
        data01_MorganDescr=data01_m.replace(r'[^\w\s]|_', '', regex=True)
        min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = min_max_scaler.fit_transform(data01_MorganDescr)
        df = pd.DataFrame(data_scaled)
        #print(df)
        df.columns = ['A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','EXP']
        
        
        ## merge data - data01 and machine learning model 1
        data01_mm = pd.concat([reverse_fingerprints,ohe_df01, df], axis=1).reset_index(drop=True)
        data01_mmm=data01_mm.replace(r'[^\w\s]|_', '', regex=True)


        return (data01_mmm, ohe_ions01, reverse_fingerprints)


def normalizedata(x, y, splitRatio, state , transform = False, property = None):
    """
    params: x & y are input and target
    return: if True, normalize x and y else split only.
    """
    # split data
    X01_train, X01_test,y01_train, y01_test = train_test_split(x, y, random_state = state, test_size = splitRatio, shuffle=True)
    
    # transform data
    if transform == True:
        ## create a scaler
        
        if property == 'Y':
            ## normalize data
            scaler = MinMaxScaler(feature_range=(-1, 1))
            # fit data
            scaler.fit(y01_train)
            # transform data
            y01_train = scaler.transform(y01_train)
            y01_test  = scaler.transform(y01_test)
            # reshape to dataframe
            y01_train = pd.DataFrame(y01_train, columns = y.columns)
            y01_test = pd.DataFrame(y01_test, columns = y.columns)
            # set index
            y01_train = y01_train.set_index(X01_train.index)
            y01_test  = y01_test.set_index(X01_test.index)
        
        if property == 'X':
            ## normalize data
            scaler = MinMaxScaler(feature_range=(-1, 1)) # convert to range(-1, 1)
            # fit data
            scaler.fit(X01_train)
            # transform data
            X01_train = scaler.transform(X01_train)
            X01_test  = scaler.transform(X01_test)
            # reshape to dataframe
            X01_train = pd.DataFrame(X01_train, columns = x.columns)
            X01_test = pd.DataFrame(X01_test, columns = x.columns)
            # set index
            X01_train = X01_train.set_index(y01_train.index)
            X01_test  = X01_test.set_index(y01_test.index)
    
    
    return (X01_train, X01_test, y01_train, y01_test)

# Test the output
def score(model, x_train, y):
    
    rsquared= model.score(x_train,y)
    y_predicted = model.predict(x_train)
    mse= mean_squared_error(y, y_predicted)
    mae=mean_absolute_error(y, y_predicted)
    
    return (mse, mae, rsquared)

def shap_plot(model,x_train,y_train,x_test, y_test,X):
    explainersvr = shap.KernelExplainer(model.predict, x_train)
    shap_valuessvr = explainersvr.shap_values(x_test)
    shap.summary_plot(shap_valuessvr, features=X, feature_names=X.columns, plot_type='bar',color='#92c5de', show = False)
    plt.gcf().set_size_inches(5,6)
    plt.rcParams.update({'font.size': 50})
    plt.show()

def plot(model,x_train, y_train,x_test, y_test):
    
    y_predicted_train = model.predict(x_train)
    y_predicted_test = model.predict(x_test)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    ax.tick_params(axis="y", which="major",right=False,direction="in",length=5)
    ax.tick_params(axis="x", which="major",direction="in",length=5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.)  # change width
    plt.legend(loc="upper left",frameon=False)
    plt.title(model,size=8)
    ax.set_xlabel("Activity coefficient $_{(measured)}$",size=12, color = 'black')
    ax.set_ylabel("Activity coefficient $_{(predicted)}$",size=12, color = 'black')
    ax.plot(y_train,y_train,color='black',linewidth=1.5)
    plt.scatter(y_train, y_predicted_train,facecolors='teal',edgecolors='teal',s=25, label="Train",marker = 'x')
    plt.scatter(y_test, y_predicted_test,facecolors='red',edgecolors='gold',s=25, label="Test",marker = 'x')
 


    #plt.savefig('SVR_PCA.png', dpi=600)

def transformPCA_ONE(x, length, type):
    """ params: dataframe of the inputs, usually train data
        returns: image of variance against number of components"""

    (n_samples, n_features) = x.shape
    num_components = min(n_samples, n_features)
    # apply PCA
    if type == 'MDFP':
        pca_x = PCA(n_components = num_components, random_state=0)
        pca_x.fit(x.iloc[:, :length])
    if type == "ALL":
        pca_x = PCA(n_components = num_components, random_state=0)
        pca_x.fit(x)
        
    #  plot
    fig, axes = plt.subplots(1, 1, figsize=(5, 3), dpi = 100)
    #
    sns.set(style='whitegrid')
        
    axes.plot(np.cumsum(pca_x.explained_variance_ratio_))
    axes.set_xlabel('Components', fontsize = 15); axes.set_ylabel('Variance', fontsize = 15)
    axes.tick_params(axis='x', labelsize = 13); axes.tick_params(axis='y', labelsize = 13)
    
    #axes.axvline(linewidth = 1, color = 'r', linestyle = '--', x = 10, ymin = 0, ymax = 1)
    
    axes.grid(False)
    axes.set_xlim([0, 30])
    #
    #plt.savefig(f'./reports/figures/mainPAPER/{filename}.png')
    plt.show()


def convertInputs(X_train, X_test,mf, type):    
    
    if type == 'MDFP':
        # fit
        pca = PCA(n_components=10) 
        length_MDFP = mf.shape[1]
        pca.fit(X_train.iloc[:, :length_MDFP])
        # predict
        pca_X01_MF_train_onlyMDFP = pca.transform(X_train.iloc[:, :length_MDFP])
        pca_X01_MF_test_onlyMDFP  = pca.transform(X_test.iloc[:, :length_MDFP])
        # convert to pandas
        pca_X01_MF_train_onlyMDFP_df = pd.DataFrame(data = pca_X01_MF_train_onlyMDFP, columns=[f'pca_{i}' for i in range(len(pca_X01_MF_train_onlyMDFP[0]))])
        pca_X01_MF_test_onlyMDFP_df = pd.DataFrame(data = pca_X01_MF_test_onlyMDFP, columns=[f'pca_{i}' for i in range(len(pca_X01_MF_test_onlyMDFP[0]))])
        # merge frames
        pca_X01_MF_train = pd.concat([pca_X01_MF_train_onlyMDFP_df, X_train.iloc[:, length_MDFP:].reset_index()], axis = 1)
        pca_X01_MF_test = pd.concat([pca_X01_MF_test_onlyMDFP_df, X_test.iloc[:, length_MDFP:].reset_index()], axis = 1)
        # drop the index column
        pca_X01_MF_train.drop(columns= ['index'], inplace=True)
        pca_X01_MF_test.drop(columns= ['index'], inplace=True)

        return pca_X01_MF_train, pca_X01_MF_test
    
    if type == 'All':
        # fit
        pca = PCA(n_components=10)
        pca.fit(X_train)
        # predict
        pca_X01_MF_train_ = pca.transform(X_train)
        pca_X01_MF_test_ = pca.transform(X_test)
        # convert to pandas
        pca_X01_MF_train = pd.DataFrame(data = pca_X01_MF_train_, columns=[f'pca_{i}' for i in range(len(pca_X01_MF_train_[0]))])
        pca_X01_MF_test = pd.DataFrame(data = pca_X01_MF_test_, columns=[f'pca_{i}' for i in range(len(pca_X01_MF_test_[0]))])
        
        return pca_X01_MF_train, pca_X01_MF_test
