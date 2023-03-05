import os
import sys
import numpy as np
import time
import rdkit
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import AllChem

path = '/home/qilintong/pdbbind'

def types_from_file(f,holo_f):
    count = 0
    distance = 4
    for line in f:
        atom_nps=[]
        prot=line.split()[0]
        ligand_file=os.path.join(path,prot,prot+'_ligand.sdf')
        try:
            mol = Chem.MolFromMolFile(ligand_file)
            ligand =AllChem.GetMorganFingerprintAsBitVect(mol, nBits=2048, radius=2)
            c=mol.GetConformer()
            count+=1
            atom_np=c.GetPositions()
            atom_nps.append(atom_np)
            centers = np.loadtxt(path + "/" + prot +"/"+prot+ "_protein_nowat_out/pockets/" + "bary" + "_centers.txt")
            if centers.shape[0]==4 and len(centers.shape)==1:
                centers=np.expand_dims(centers,axis=0)
            limit = centers.shape[0]
            if not (centers.shape[0]==0 and len(centers.shape)==1):
                sorted_centers = centers[:int(limit), 1:]
                for i in range(int(limit)):
                    label=0
                    for atom_np in atom_nps:
                        dist = np.linalg.norm((atom_np - sorted_centers[i,:]), axis=1)
                        rel_centers = np.where(dist <= float(distance), 1, 0)
                        if (np.count_nonzero(rel_centers) > 0):
                            label =1
                        else:
                            label =0
                        holo_f.write(str(label)+ ' '+ str(sorted_centers[i][0])+ ' '+ str(sorted_centers[i][1])+ ' '+ str(sorted_centers[i][2])+ ' '+prot+'/'+prot+'_protein_nowat.gninatypes\n')
        except:
              pass
              continue
    print(count)
train_prots=open('trainval5.txt','r').readlines()
holo_train=open('pdbbind_train_5.types','w')
types_from_file(train_prots,holo_train)
test_prots=open('test5.txt','r').readlines()
holo_test=open('pdbbind_test_5.types','w')
types_from_file(test_prots,holo_test)
