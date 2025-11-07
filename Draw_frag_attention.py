import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt

# paint the figure of the attention value of the corresponding , the figure will be saved in attention_picture folder

path = os.path.join('.//dataset//MG_plus_reference.csv')
data_from = os.path.realpath(path)
df = pd.read_csv(data_from)
pattern = np.array([df['First-Order Group'], df['SMARTs'], df['Priority']])

sorted_pattern = pattern[:, np.argsort(pattern[2, :])]
frag_name_list = list(dict.fromkeys(sorted_pattern[0, :]))
frag_dim = len(frag_name_list)

sorted_pattern = pattern[:,np.argsort(pattern[2,:])]

colors = (0.8,0.8,0)
attention_list = [0.02895916700363159]
smiles = 'C1=COC(=C1)CO'
mol = Chem.MolFromSmiles(smiles)
pat_list = []
prior_set = set()
mol_size = mol.GetNumAtoms()

hit_ats = []
hit_bonds = []
atom_cols = {}
bond_cols = {}
z = 0
for j, patt in enumerate(sorted_pattern[1,:]):
    pat = Chem.MolFromSmarts(patt)
    frags = list(mol.GetSubstructMatches(pat))
    if len(frags):
        # print(frags,sorted_pattern[0][j])
        for i, item in enumerate(frags):
            item_set = set(item)
            new_frags = frags[:i] + frags[i + 1:]
            left_set = set(sum(new_frags,()))
            if not item_set.isdisjoint(left_set):
                frags = new_frags

        for _, frag in enumerate(frags):
            frag_set = set(frag)
            if prior_set.isdisjoint(frag_set):
                ats = frag_set
            else:
                ats = {}
            if ats:
                print(sorted_pattern[1,:][j])
                for i in ats:
                    hit_ats.append(i)
                for i in ats:
                    color = []
                    color.append(colors[0]*float(attention_list[0]))
                    color.append(colors[1]*float(attention_list[0]))
                    color.append(colors[2]*float(attention_list[0]))
                    atom_cols[i] = tuple(color)
                prior_set.update(ats)


def draw_molecule_from_smiles(smiles, image_size=(400, 400), filename=None):

    try:
        # 从SMILES字符串创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            print("错误：无法解析SMILES字符串")
            return None, None
        
        # 方法1：直接绘制，不使用MolDrawOptions
        img = Draw.MolToImage(mol, size=image_size)
        
        # 保存或显示图像
        if filename:
            img.save(filename)
            print(f"分子结构已保存到: {filename}")
        else:
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'分子结构: {smiles}')
            plt.show()
        
        return mol, img
        
    except Exception as e:
        print(f"发生错误: {e}")
        return None, None

# 调用函数
mol, img = draw_molecule_from_smiles(smiles=smiles, filename='attention_picture.png')


