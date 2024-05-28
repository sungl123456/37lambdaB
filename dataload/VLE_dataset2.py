
import os
import dgl.backend as F
import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import save_graphs, load_graphs
from utils.mol2graph import graph_2_frag, create_channels

class VLE_Dataset2(object):
    def __init__(self,  df, params, name, smiles_2_graph,
                 atom_featurizer, bond_featurizer, mol_featurizer, 
                 cache_file_path, load=False, log_every=100, error_log=None, fragmentation=None):
        self.df = df
        self.name = name
        self.cache_file_path = cache_file_path
        self._prepare(params, smiles_2_graph,atom_featurizer,bond_featurizer,mol_featurizer,load,log_every,error_log)
        self.whe_frag = False
        if fragmentation is not None:
            self.whe_frag = True
            self._prepare_frag(params, fragmentation, load, log_every, error_log)
            self._prepare_channel()
        
    def