# 该文件的主要作用是将文献中与沸点临界温度数据对应的SMILES式规范化，并去除重复项
# This file is applied for standadize SMILES corresponding to boiling point and critical 
# temperature data extracted from the literature, while concurrently eliminating redundant entries.

# python Tb_Tc_normalized Tb_JCIM.csv
# >> Tb_JCIM_normalized.csv

from rdkit import Chem
import sys
import pandas as pd
import os
import numpy as np

filename = sys.argv[1]
file  = open(filename,'r')

path = os.path.join(filename)
df = pd.read_csv(path,sep = ',')
smi_list = []
value_list = []
for smile in df['smiles']:
    mol = Chem.MolFromSmiles(smile)
    can_smiles = Chem.MolToSmiles(mol, isomericSmiles = True)
    index = df[(df['smiles'] == smile)]
    index = np.array(index)
    if can_smiles in smi_list:
        continue
    else:
        smi_list.append(can_smiles)
        value_list.append(index[0][1])
dot_index = filename.find('.')
filename_new = filename[:dot_index]
file_write = open(filename_new+'_normalized.csv','w')
file_write.write('smiles'+','+'value'+'\n')
for i in range(len(smi_list)):
    file_write.write(smi_list[i]+','+str(value_list[i])+'\n')

file_write.close()
file.close()