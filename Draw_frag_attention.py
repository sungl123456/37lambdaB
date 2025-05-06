import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

# paint the figure of the attention value of the corresponding , the figure will be saved in attention_picture folder

path = os.path.join('.//dataset//MG_plus_reference.csv')
data_from = os.path.realpath(path)
df = pd.read_csv(data_from)
pattern = np.array([df['First-Order Group'], df['SMARTs'], df['Priority']])

sorted_pattern = pattern[:, np.argsort(pattern[2, :])]
frag_name_list = list(dict.fromkeys(sorted_pattern[0, :]))
frag_dim = len(frag_name_list)

sorted_pattern = pattern[:,np.argsort(pattern[2,:])]

colors = (0.8,0,0)
attention_list = [0.02895916700363159, 0.8896619439125061, 0.04207463264465332]
smiles = 'CCO'
mol = Chem.MolFromSmiles(smiles)
pat_list = []
prior_set = set()
mol_size = mol.GetNumAtoms()
picture_path = 'attention_picture'
if not os.path.exists(picture_path):
    os.mkdir(picture_path)
path3 = os.path.join(os.path.realpath(picture_path), smiles +'_Tb.png')
d = rdMolDraw2D.MolDraw2DCairo(500, 500)

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
                #这里未来还是要考虑把化学键加进来
                # if len(frag) > 1:
                    # for bond in pat.GetBonds():
                    #     aid1 = frags[bond.GetBeginAtomIdx()]
                    #     aid2 = frags[bond.GetEndAtomIdx()]
                    #     hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx)
                for i in ats:
                    hit_ats.append(i)
                # print(hit_ats,prior_set,sorted_pattern[0,:][j])
                # for i, at in enumerate(hit_ats):
                for i in ats:
                    color = []
                    color.append(colors[0]*float(attention_list[0]*5))
                    color.append(colors[1]*float(attention_list[0]*5))
                    color.append(colors[2]*float(attention_list[0]*5))
                    atom_cols[i] = tuple(color)
                prior_set.update(ats)
                attention_list.pop(0)
                # for i, bd in enumerate(hit_bonds):
                #     bond_cols[bd] = colors    
     
rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms = hit_ats, highlightAtomColors = atom_cols)#, highlightbonds = hit_bonds,  highlightBondColors = bond_cols)
hit_ats = []
# hit_bonds = []
atom_cols = {}
# bond_cols = {}

d.FinishDrawing()
d.WriteDrawingText(path3)


