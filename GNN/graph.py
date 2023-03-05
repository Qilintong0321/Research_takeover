import os
import sys
import numpy as np
import time
import rdkit
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import AllChem
from process_mols import  get_lig_graph_revised

path = '/home/qilintong/pdbbind/1a5v/1.pdb'
mol = Chem.MolFromPDBFile(path)
lig_graph=get_lig_graph_revised(mol)
lig_graph=lig_graph.to('cuda')
print(lig_graph.ndata['feat'].device)
print(lig_graph)
