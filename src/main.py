import os
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors,PandasTools
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_curve, roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold, train_test_split

import vectorize as vec
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

#############################
in_0 = '../data/newout0-tab.tsv'
in_1 = '../data/newout1-tab.tsv'
original_0 = '../data/w0-tab.tsv'
original_1 = '../data/w1-tab.tsv'
wash = '../data/washed-nr-er.tsv'
#############################

def ToMols(file):
    mols = pd.read_table(file)
    Chem.PandasTools.AddMoleculeColumnToFrame(mols, smilesCol='washed_mol', molCol='ROMol')
    bad_mol_idx = mols['ROMol'].isnull()
    mols = mols[~bad_mol_idx]

    names = [x[0] for x in Descriptors._descList]
    print("Number of descriptors in the rdkit: ", len(names))

    #desc_for_now = ['TPSA','SlogP_VSA1','EState_VSA1','SMR_VSA1','MolLogP','MolMR','BalabanJ','HallKie rAlpha','Kappa1','Kappa2','Kappa3','RingCount','NumHAcceptors','NumHDonors']
    desc_for_now = names
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_for_now)

    desc = OrderedDict()
    for mol in mols.index:
        desc[mol] = calculator.CalcDescriptors(mols.loc[mol, 'ROMol'])

    desc_mols = pd.DataFrame.from_dict(desc, orient='index', columns=desc_for_now)
    use_list = ['BertzCT', 'BalabanJ', 'MolLogP', 'Chi4v', 'MaxAbsEStateIndex', 'VSA_EState6', 'VSA_EState5', 'VSA_EState4',
                'PEOE_VSA7', 'SMR_VSA7', 'VSA_EState7', 'SlogP_VSA6', 'EState_VSA4', 'fr_bicyclic', 'EState_VSA7']

    remove_list = (set(desc_mols.columns) - set(use_list))
    desc_mols = desc_mols.drop(remove_list, axis=1)
    # for i, cont in denumerate(desc_mols):
    #     for col in desc_mols.columns:
    #         print('desc_mols.column:  ', col)
    #         if np.isnan(desc_mols[i, col]):
    #             pass
    desc_mols = desc_mols.dropna()
    print(desc_mols[:3])
    sys.exit()
    return desc_mols, mols

if __name__=='__main__':

    with open(in_0, 'r') as f:
        tmp = f.read().split()
    
    #desc_mols0, mols0 = ToMols(in_0)
    # desc_mols1, mols1 = ToMols(in_1)
    desc_wash_mols, wash_mols = ToMols(wash)
    
    x_train, x_test, y_train, y_test = train_test_split(desc_wash_mols, wash_mols.Active, train_size=0.75, test_size=0.25, random_state=42)
    # x_train = pd.concat([x_train, desc_mols0], axis=0)
    # tmp = pd.Series([0 for _ in range(len(desc_mols0))])
    # y_train = pd.concat([y_train, tmp], axis=0)
    # x_train = pd.concat([x_train, desc_mols1], axis=0)
    # tmp = pd.Series([1 for _ in range(len(desc_mols1))])
    # y_train = pd.concat([y_train, tmp], axis=0)
    
    model = RandomForestClassifier(n_estimators = 100)
    model.fit(x_train, y_train)
    #print("0_Accuracy  on  training set: {:.3f}".format(model.score(x_train, y_train)))
    #print("0_Accuracy  on  testing set: {:.3f}".format(model.score(x_test, y_test)))
    df = pd.DataFrame(model.feature_importances_, index = x_train.columns)
    
    prob = model.predict(x_test)
    accuracy = accuracy_score(y_test, prob)
    precision = precision_score(y_test, prob)
    recall = recall_score(y_test, prob)
    f1 = f1_score(y_test, prob)
    matthews = matthews_corrcoef(y_test, prob)
    
    print('Acc: {}\nPrec: {}\nRec: {}\nF1: {}\nmatthews: {}'.format(accuracy, precision, recall, f1, matthews))
