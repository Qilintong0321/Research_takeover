from Bio.PDB import PDBParser, PDBIO, Select
import Bio
import os
import sys
import re
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import molgrid
import argparse
import time
from skimage.morphology import binary_dilation
from skimage.morphology import cube
from skimage.morphology import closing
from skimage.segmentation import clear_border
from skimage.measure import label
import struct
from clean_pdb import clean_pdb
from get_centers import get_centers
from types_and_gninatyper import gninatype,create_types
from model import Model
from rank_pockets import test_model
from unet import Unet
from segment_pockets import test
import gc
if __name__ == '__main__':
    #clean pdb file and remove hetero atoms/non standard residues
    protein_file=sys.argv[1]
    protein_nowat_file=protein_file.replace('.pdb','_nowat.pdb')
    os.system('fpocket -f '+protein_nowat_file)
    fpocket_dir=os.path.join(protein_nowat_file.replace('.pdb','_out'),'pockets')
    get_centers(fpocket_dir)
    barycenter_file=os.path.join(fpocket_dir,'bary_centers.txt')
    #types and gninatyper
    protein_gninatype=gninatype(protein_nowat_file)
    class_types=create_types(barycenter_file,protein_gninatype)
