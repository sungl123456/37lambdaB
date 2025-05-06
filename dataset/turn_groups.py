
from rdkit import Chem


mol = Chem.MolFromSmiles('SS')
SMARTS = Chem.MolToSmarts(mol)
print(SMARTS)

# mol = Chem.MolFromSmarts('[2H]-[#8]-[2H]')
# SMILES = Chem.MolToSmiles(mol)
# print(SMILES)