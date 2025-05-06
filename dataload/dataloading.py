import os
import platform

import numpy as np
import pandas as pd

from rdkit import Chem

def import_dataset(params):
    dataset_name = params['Dataset']
    path = os.path.join('.//dataset',dataset_name + '.csv')
    df = pd.read_csv(path,sep = ',')
    if dataset_name == 'azeotrope_classification' or dataset_name == 'azeotrope_classification_without_water' or dataset_name == 'azeotrope_classification_water':
        df['comp1'] = look_for_smiles(df['comp1'])
        df['comp2'] = look_for_smiles(df['comp2'])
        df['value'] = df['value'].astype(int)
        return df
    if dataset_name == 'VLE_azeotrope' or dataset_name =='VLE_zeotrope' or dataset_name == 'VLE_azeotrope_with_water' or dataset_name == 'VLE_zeotrope_with_water' or dataset_name == 'VLE_total':
        scaling_T = minmaxNormalization(df['T'])
        scaling_y1 = minmaxNormalization(df['y1'])
        df['comp1'] = look_for_smiles(df['comp1'])
        df['comp2'] = look_for_smiles(df['comp2'])
        df['T'] = scaling_T.Scaler(df['T'])
        df['y1'] = scaling_y1.Scaler(df['y1'])
        return df, scaling_T, scaling_y1
    else:
        scaling = minmaxNormalization(df['value'])
        df['value'] = scaling.Scaler(df['value'])
        return df, scaling

class Standardization:
    def __init__(self,input):
        n = len(input)
        self.average = np.sum(input) * (1 / n)
        self.varience = np.sqrt((1 / n) * np.sum((input-self.average)**2))
    
    def Scaler(self, input):
        return (input - self.average) / self.varience
    
    def ReScaler(self, input):
        return self.average + self.varience * input

class minmaxNormalization:
    def __init__(self, input):
        self.min = np.min(input)
        self.max = np.max(input)

    def Scaler(self, input):
        return (input - self.min) / (self.max - self.min)

    def ReScaler(self, input):
        return self.min + input * (self.max - self.min)


def get_canonical_smiles(smiles):

    smi_list = []
    for s in smiles:
        try:
            smi_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
        except:
            print('Failed to generate the canonical smiles from ',s ,'. Please check the inputs.')
        
    return smi_list

def look_for_smiles(names):
    smi_list = []
    path = os.path.join('.//dataset','smiles.csv')
    comp = pd.read_csv(path,usecols = ['name','smiles'])

    for name in names:
        df = comp[(comp['name'] == name)]
        df = np.array(df)

        if len(df) == 0:
            print(f"the {name} is not involved in the dataset, please input the SMILES directly")
            smile = input()
            smi_list.append(df[0])
        else:
            smile = df[0][1]
            smi_list.append(smile)
    return smi_list

def look_for_Tb(names):
    Tb_list = []
    path_Tb = os.path.join('.//dataset','Tb_VLE.csv')
    Tb = pd.read_csv(path_Tb,usecols = ['smiles','value'])

    for name in names:
        df = Tb[(Tb['smiles'] == name)]
        df = np.array(df)

        if len(df) == 0:
            print(f"the {name} is not involved in the dataset, please input the Tb directly")
            Tb_input = input()
            Tb_list.append(df[0])
        else:
            Tb_input = df[0][1]
            Tb_list.append(Tb_input)
    return Tb_list

def look_for_Tc(names):
    Tc_list = []
    path_Tb = os.path.join('.//dataset','Tc_VLE.csv')
    Tc = pd.read_csv(path_Tb,usecols = ['smiles','value'])

    for name in names:
        df = Tc[(Tc['smiles'] == name)]
        df = np.array(df)
        if len(df) == 0:
            print(f"the {name} is not involved in the dataset, please input the Tc directly")
            Tc_input = input()
            Tc_list.append(df[0])
        else:
            Tc_input = df[0][1]
            Tc_list.append(Tc_input)
    return Tc_list
