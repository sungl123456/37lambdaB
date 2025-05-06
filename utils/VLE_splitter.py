
import math
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data import Subset

def Azeotrope_Splitter(comp, seed, train_size, val_size, all_size):
    rand_state = np.random.RandomState(int(seed))
    indices = [*range(all_size)]
    rand_state.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:val_size]
    test_indices = indices[val_size:all_size]

    comp_train = []
    comp_val = []
    comp_test = []
    for i in train_indices:
        comp_train.append(comp[i])
    for i in val_indices:
        comp_val.append(comp[i])
    for i in test_indices:
        comp_test.append(comp[i])
    
    comp_train = pd.DataFrame(comp_train,columns=['comp1','comp2','value'])
    comp_val = pd.DataFrame(comp_val,columns=['comp1','comp2','value'])
    comp_test = pd.DataFrame(comp_test,columns=['comp1','comp2','value'])
    comp_all = pd.DataFrame(comp,columns=['comp1','comp2','value'])
    return comp_train, comp_val, comp_test, comp_all

def pair_Splitter(comp, seed, all_size):
    mixtures = []

    for i in range(all_size):
        mixture = []
        mixture.append(comp[i][0])
        mixture.append(comp[i][1])
        if mixture not in mixtures and [mixture[1],mixture[0]] not in mixtures:
            mixtures.append(mixture)
            mixtures.append([mixture[1],mixture[0]])

    mixture_size = len(mixtures)
    rand_state = np.random.RandomState(int(seed))
    indices = [*range(mixture_size)]
    rand_state.shuffle(indices)
    train_size = math.floor(mixture_size*0.8)
    val_size = math.floor(mixture_size * 0.9)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:val_size]
    test_indices = indices[val_size:mixture_size]

    comp_train = []
    comp_val = []
    comp_test = []
    for i in train_indices:
        for j in range(all_size):
            if comp[j][0] == mixtures[i][0] and comp[j][1] == mixtures[i][1]:
                comp_train.append(comp[j])
        for j in range(all_size):
            if comp[j][0] == mixtures[i][1] and comp[j][1] == mixtures[i][0]:
                comp_train.append(comp[j])

    for i in val_indices:
        for j in range(all_size):
            if comp[j][0] == mixtures[i][0] and comp[j][1] == mixtures[i][1]:
                comp_val.append(comp[j])
        for j in range(all_size):
            if comp[j][0] == mixtures[i][1] and comp[j][1] == mixtures[i][0]:
                comp_val.append(comp[j])
    for i in test_indices:
        for j in range(all_size):
            if comp[j][0] == mixtures[i][0] and comp[j][1] == mixtures[i][1]:
                comp_test.append(comp[j])
        for j in range(all_size):
            if comp[j][0] == mixtures[i][1] and comp[j][1] == mixtures[i][0]:
                comp_test.append(comp[j])

    comp_train = pd.DataFrame(comp_train,columns = ['comp1','comp2','value'])
    comp_val = pd.DataFrame(comp_val,columns = ['comp1','comp2','value'])
    comp_test = pd.DataFrame(comp_test,columns = ['comp1','comp2','value'])
    comp_all = pd.DataFrame(comp,columns = ['comp1','comp2','value'])
    return comp_train, comp_val, comp_test, comp_all

def pair_Splitter2(comp, seed, all_size):

    mixtures = []
    for i in range(all_size):
        mixture = []
        mixture.append(comp[i][0])
        mixture.append(comp[i][1])
        if mixture not in mixtures and [mixture[1],mixture[0]] not in mixtures:
            mixtures.append(mixture)
            

    mixture_size = len(mixtures)
    rand_state = np.random.RandomState(int(seed))
    indices = [*range(mixture_size)]
    rand_state.shuffle(indices)
    train_size = math.floor(mixture_size*0.8)
    val_size = math.floor(mixture_size * 0.9)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:val_size]
    test_indices = indices[val_size:mixture_size]

    comp_train = []
    comp_val = []
    comp_test = []
    for i in train_indices:
        for j in range(all_size):
            if comp[j][0] == mixtures[i][0] and comp[j][1] == mixtures[i][1]:
                comp_train.append(comp[j])
        for j in range(all_size):
            if comp[j][0] == mixtures[i][1] and comp[j][1] == mixtures[i][0]:
                comp_train.append(comp[j])

    for i in val_indices:
        for j in range(all_size):
            if comp[j][0] == mixtures[i][0] and comp[j][1] == mixtures[i][1]:
                comp_val.append(comp[j])
        for j in range(all_size):
            if comp[j][0] == mixtures[i][1] and comp[j][1] == mixtures[i][0]:
                comp_val.append(comp[j])
    for i in test_indices:
        for j in range(all_size):
            if comp[j][0] == mixtures[i][0] and comp[j][1] == mixtures[i][1]:
                comp_test.append(comp[j])
        for j in range(all_size):
            if comp[j][0] == mixtures[i][1] and comp[j][1] == mixtures[i][0]:
                comp_test.append(comp[j])

    comp_train = pd.DataFrame(comp_train,columns=['comp1','comp2','value','Tb_comp1','Tb_comp2','Tc_comp1','Tc_comp2'])
    comp_val = pd.DataFrame(comp_val,columns=['comp1','comp2','value','Tb_comp1','Tb_comp2','Tc_comp1','Tc_comp2'])
    comp_test = pd.DataFrame(comp_test,columns=['comp1','comp2','value','Tb_comp1','Tb_comp2','Tc_comp1','Tc_comp2'])
    comp_all = pd.DataFrame(comp,columns=['comp1','comp2','value','Tb_comp1','Tb_comp2','Tc_comp1','Tc_comp2'])
    return comp_train, comp_val, comp_test, comp_all


class pair_Splitter3(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        
    def Random_Splitter(self, comp, seed = 100, frac_train = 0.8, frac_val = 0.9):
        self.length = len(comp)
        self.comp = comp
        mixtures = []              
        for i in range(self.length):
            mixture = []
            mixture.append(comp[i][0])
            mixture.append(comp[i][1])
            if mixture not in mixtures and [mixture[1],mixture[0]] not in mixtures:
                mixtures.append(mixture)
        
        self.mixtures = mixtures
        self.index = len(self.mixtures)

        rand_state = np.random.RandomState(int(seed))
        indices = [*range(self.index)]
        rand_state.shuffle(indices)

        num_train = int(frac_train*self.index)
        num_val  = int(frac_val*self.index)
        
        train_index = []
        val_index = []
        test_index = []
        for i in range(num_train):
            for j in range(self.length):
                if mixtures[indices[i]] == [comp[j][0],comp[j][1]]:
                    train_index.append(j)
                if [mixtures[indices[i]][1],mixtures[indices[i]][0]] == [comp[j][0],comp[j][1]]:
                    train_index.append(j)
        for i in range(num_train,num_val):
            for j in range(self.length):
                if mixtures[indices[i]] == [comp[j][0],comp[j][1]]:
                    val_index.append(j)
                if [mixtures[indices[i]][1],mixtures[indices[i]][0]] == [comp[j][0],comp[j][1]]:
                    val_index.append(j)
        for i in range(num_val,self.index):
            for j in range(self.length):
                if mixtures[indices[i]] == [comp[j][0],comp[j][1]]:
                    test_index.append(j)
                if [mixtures[indices[i]][1],mixtures[indices[i]][0]] == [comp[j][0],comp[j][1]]:
                    test_index.append(j)

        train_dataset = Subset(self.dataset, train_index)
        val_dataset = Subset(self.dataset, val_index)
        test_dataset = Subset(self.dataset, test_index)

        return  train_dataset, val_dataset, test_dataset, self.dataset

def VLE_Splitter(comp, seed, all_size):
    
    mixtures = []

    for i in range(all_size):
        mixture = []
        mixture.append(comp[i][0])
        mixture.append(comp[i][1])
        if mixture not in mixtures and [mixture[1],mixture[0]] not in mixtures:
            mixtures.append(mixture)
            mixtures.append([mixture[1],mixture[0]]) 

    mixture_size = len(mixtures)
    rand_state = np.random.RandomState(int(seed))
    indices = [*range(mixture_size)]
    rand_state.shuffle(indices)
    train_size = math.floor(mixture_size*0.8)
    val_size = math.floor(mixture_size * 0.9)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:val_size]
    test_indices = indices[val_size:mixture_size]

    comp_train = []
    comp_val = []
    comp_test = []
    for i in train_indices:
        for j in range(all_size):
            if comp[j][0] == mixtures[i][0] and comp[j][1] == mixtures[i][1]:
                comp_train.append(comp[j])
        for j in range(all_size):
            if comp[j][0] == mixtures[i][1] and comp[j][1] == mixtures[i][0]:
                comp_train.append(comp[j])

    for i in val_indices:
        for j in range(all_size):
            if comp[j][0] == mixtures[i][0] and comp[j][1] == mixtures[i][1]:
                comp_val.append(comp[j])
        for j in range(all_size):
            if comp[j][0] == mixtures[i][1] and comp[j][1] == mixtures[i][0]:
                comp_val.append(comp[j])
    for i in test_indices:
        for j in range(all_size):
            if comp[j][0] == mixtures[i][0] and comp[j][1] == mixtures[i][1]:
                comp_test.append(comp[j])
        for j in range(all_size):
            if comp[j][0] == mixtures[i][1] and comp[j][1] == mixtures[i][0]:
                comp_test.append(comp[j])


    comp_train = pd.DataFrame(comp_train,columns=['comp1','comp2','x1','T','y1','Tb_comp1','Tb_comp2','Tc_comp1','Tc_comp2'])
    comp_val = pd.DataFrame(comp_val,columns=['comp1','comp2','x1','T','y1','Tb_comp1','Tb_comp2','Tc_comp1','Tc_comp2'])
    comp_test = pd.DataFrame(comp_test,columns=['comp1','comp2','x1','T','y1','Tb_comp1','Tb_comp2','Tc_comp1','Tc_comp2'])
    comp_all = pd.DataFrame(comp,columns=['comp1','comp2','x1','T','y1','Tb_comp1','Tb_comp2','Tc_comp1','Tc_comp2'])

    return  comp_train, comp_val, comp_test, comp_all

class VLE_Splitter2(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        
    def Random_Splitter(self, comp, seed = 100, frac_train = 0.8, frac_val = 0.9):
        self.length = len(comp)
        self.comp = comp
        mixtures = []              
        for i in range(self.length):
            mixture = []
            mixture.append(comp[i][0])
            mixture.append(comp[i][1])
            if mixture not in mixtures and [mixture[1],mixture[0]] not in mixtures:
                mixtures.append(mixture)

        self.mixtures = mixtures
        self.index = len(self.mixtures)

        rand_state = np.random.RandomState(int(seed))
        indices = [*range(self.index)]
        rand_state.shuffle(indices)

        num_train = int(frac_train*self.index)
        num_val  = int(frac_val*self.index)

        train_index = []
        val_index = []
        test_index = []
        for i in range(num_train):
            for j in range(self.length):
                if mixtures[indices[i]] == [comp[j][0],comp[j][1]] or[mixtures[indices[i]][1],mixtures[indices[i]][0]] == [comp[j][0],comp[j][1]]:
                    train_index.append(j)

        for i in range(num_train,num_val):
            for j in range(self.length):
                if mixtures[indices[i]] == [comp[j][0],comp[j][1]] or[mixtures[indices[i]][1],mixtures[indices[i]][0]] == [comp[j][0],comp[j][1]]:
                    val_index.append(j)

        for i in range(num_val,self.index):
            for j in range(self.length):
                if mixtures[indices[i]] == [comp[j][0],comp[j][1]] or[mixtures[indices[i]][1],mixtures[indices[i]][0]] == [comp[j][0],comp[j][1]]:
                    test_index.append(j)

        train_dataset = Subset(self.dataset, train_index)
        val_dataset = Subset(self.dataset, val_index)
        test_dataset = Subset(self.dataset, test_index)

        return  train_dataset, val_dataset, test_dataset, self.dataset