import time
import dgl
import torch
from .metrics import Metrics, Metrics_classification
from .Set_Seed_Reproducibility import set_seed


class Azeotrope_PreFetch(object):
    def __init__(self, train_loader, val_loader, test_loader, raw_loader, frag):
        if frag == 4:
            self.train_batched_origin_graph_list_comp1, self.train_batched_origin_graph_list_comp2 = [], []
            self.train_batched_frag_graph_list_comp1, self.train_batched_frag_graph_list_comp2 = [], []
            self.train_batched_motif_graph_list_comp1, self.train_batched_motif_graph_list_comp2 = [], [] 
            self.train_smiles_list_comp1,self.train_smiles_list_comp2 = [], [] 
            self.train_Tb_comp1, self.train_Tc_comp1,self.train_Tb_comp2, self.train_Tc_comp2 = [], [], [], []
            self.train_iter = []
            self.train_x1, self.train_T, self.train_y1 = [], [], [] 
            for iter, batch in enumerate(train_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2, _smiles1,_smiles2, _Tb_comp1, _Tc_comp1, _Tb_comp2, _Tc_comp2,_x1, _T, _y1 = batch
                self.train_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.train_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.train_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.train_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.train_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.train_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.train_smiles_list_comp1.append(_smiles1)
                self.train_smiles_list_comp2.append(_smiles2)
                self.train_Tb_comp1.append(_Tb_comp1)
                self.train_Tc_comp1.append(_Tc_comp1)
                self.train_Tb_comp2.append(_Tb_comp2)
                self.train_Tc_comp2.append(_Tc_comp2)
                self.train_iter.append(iter)
                self.train_x1.append(_x1)
                self.train_T.append(_T)
                self.train_y1.append(_y1)
            
            self.val_batched_origin_graph_list_comp1, self.val_batched_origin_graph_list_comp2 = [], []
            self.val_batched_frag_graph_list_comp1, self.val_batched_frag_graph_list_comp2 = [], []
            self.val_batched_motif_graph_list_comp1, self.val_batched_motif_graph_list_comp2 = [], []
            self.val_smiles_list_comp1,self.val_smiles_list_comp2 = [], [] 
            self.val_Tb_comp1, self.val_Tc_comp1, self.val_Tb_comp2, self.val_Tc_comp2 = [], [], [], []
            self.val_iter = []
            self.val_x1, self.val_T, self.val_y1 = [], [], [] 
            for iter, batch in enumerate(val_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2, _smiles1, _smiles2, Tb_comp1, Tc_comp1, Tb_comp2, Tc_comp2,_x1, _T, _y1 = batch
                self.val_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.val_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.val_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.val_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.val_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.val_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.val_smiles_list_comp1.append(_smiles1)
                self.val_smiles_list_comp2.append(_smiles2)
                self.val_Tb_comp1.append(Tb_comp1)
                self.val_Tc_comp1.append(Tc_comp1)
                self.val_Tb_comp2.append(Tb_comp2)
                self.val_Tc_comp2.append(Tc_comp2)
                self.val_iter.append(iter)                
                self.val_x1.append(_x1)
                self.val_T.append(_T)
                self.val_y1.append(_y1)

            self.test_batched_origin_graph_list_comp1, self.test_batched_origin_graph_list_comp2 = [], []
            self.test_batched_frag_graph_list_comp1, self.test_batched_frag_graph_list_comp2 = [], []
            self.test_batched_motif_graph_list_comp1, self.test_batched_motif_graph_list_comp2 = [], []
            self.test_smiles_list_comp1,self.test_smiles_list_comp2 = [], []
            self.test_Tb_comp1, self.test_Tc_comp1, self.test_Tb_comp2, self.test_Tc_comp2 = [], [], [], []
            self.test_iter = []
            self.test_x1, self.test_T, self.test_y1 = [], [], []
            for iter, batch in enumerate(test_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2,  _smiles1, _smiles2, Tb_comp1, Tc_comp1, Tb_comp2, Tc_comp2, _x1, _T, _y1 = batch
                self.test_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.test_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.test_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.test_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.test_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.test_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.test_smiles_list_comp1.append(_smiles1)
                self.test_smiles_list_comp2.append(_smiles2)
                self.test_Tb_comp1.append(Tb_comp1)
                self.test_Tc_comp1.append(Tc_comp1)
                self.test_Tb_comp2.append(Tb_comp2)
                self.test_Tc_comp2.append(Tc_comp2)
                self.test_iter.append(iter)
                self.test_x1.append(_x1)
                self.test_T.append(_T)
                self.test_y1.append(_y1)

            self.all_batched_origin_graph_list_comp1, self.all_batched_origin_graph_list_comp2 = [], []
            self.all_batched_frag_graph_list_comp1, self.all_batched_frag_graph_list_comp2 = [], []
            self.all_batched_motif_graph_list_comp1, self.all_batched_motif_graph_list_comp2 = [], [] 
            self.all_smiles_list_comp1,self.all_smiles_list_comp2 = [], [] 
            self.all_Tb_comp1, self.all_Tc_comp1, self.all_Tb_comp2, self.all_Tc_comp2 = [], [], [], []
            self.all_iter = []
            self.all_x1, self.all_T, self.all_y1 = [], [], []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2,  _smiles1, _smiles2, Tb_comp1, Tc_comp1, Tb_comp2, Tc_comp2, _x1, _T, _y1 = batch
                self.all_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.all_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.all_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.all_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.all_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.all_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.all_smiles_list_comp1.append(_smiles1)
                self.all_smiles_list_comp2.append(_smiles2)
                self.all_Tb_comp1.append(Tb_comp1)
                self.all_Tc_comp1.append(Tc_comp1)
                self.all_Tb_comp2.append(Tb_comp2)
                self.all_Tc_comp2.append(Tc_comp2)
                self.all_iter.append(iter)
                self.all_x1.append(_x1)
                self.all_T.append(_T)
                self.all_y1.append(_y1)



        elif frag == 3:
            self.train_batched_origin_graph_list_comp1, self.train_batched_origin_graph_list_comp2 = [], []
            self.train_batched_frag_graph_list_comp1, self.train_batched_frag_graph_list_comp2 = [], []
            self.train_batched_motif_graph_list_comp1, self.train_batched_motif_graph_list_comp2 = [], [] 
            self.train_targets_list = [] 
            self.train_smiles_list_comp1,self.train_smiles_list_comp2 = [], [] 
            self.train_Tb_comp1, self.train_Tc_comp1,self.train_Tb_comp2, self.train_Tc_comp2 = [], [], [], []
            self.train_iter = []
            for iter, batch in enumerate(train_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2, _targets, _smiles1,_smiles2, _Tb_comp1, _Tc_comp1, _Tb_comp2, _Tc_comp2 = batch
                self.train_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.train_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.train_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.train_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.train_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.train_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.train_targets_list.append(_targets)
                self.train_smiles_list_comp1.append(_smiles1)
                self.train_smiles_list_comp2.append(_smiles2)
                self.train_Tb_comp1.append(_Tb_comp1)
                self.train_Tc_comp1.append(_Tc_comp1)
                self.train_Tb_comp2.append(_Tb_comp2)
                self.train_Tc_comp2.append(_Tc_comp2)
                self.train_iter.append(iter)
            
            self.val_batched_origin_graph_list_comp1, self.val_batched_origin_graph_list_comp2 = [], []
            self.val_batched_frag_graph_list_comp1, self.val_batched_frag_graph_list_comp2 = [], []
            self.val_batched_motif_graph_list_comp1, self.val_batched_motif_graph_list_comp2 = [], [] 
            self.val_targets_list = [] 
            self.val_smiles_list_comp1,self.val_smiles_list_comp2 = [], [] 
            self.val_Tb_comp1, self.val_Tc_comp1, self.val_Tb_comp2, self.val_Tc_comp2 = [], [], [], []
            self.val_iter = []
            for iter, batch in enumerate(val_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2, _targets, _smiles1, _smiles2, Tb_comp1, Tc_comp1, Tb_comp2, Tc_comp2 = batch
                self.val_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.val_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.val_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.val_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.val_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.val_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.val_targets_list.append(_targets)
                self.val_smiles_list_comp1.append(_smiles1)
                self.val_smiles_list_comp2.append(_smiles2)
                self.val_Tb_comp1.append(Tb_comp1)
                self.val_Tc_comp1.append(Tc_comp1)
                self.val_Tb_comp2.append(Tb_comp2)
                self.val_Tc_comp2.append(Tc_comp2)
                self.val_iter.append(iter)
            
            self.test_batched_origin_graph_list_comp1, self.test_batched_origin_graph_list_comp2 = [], []
            self.test_batched_frag_graph_list_comp1, self.test_batched_frag_graph_list_comp2 = [], []
            self.test_batched_motif_graph_list_comp1, self.test_batched_motif_graph_list_comp2 = [], [] 
            self.test_targets_list = [] 
            self.test_smiles_list_comp1,self.test_smiles_list_comp2 = [], []
            self.test_Tb_comp1, self.test_Tc_comp1, self.test_Tb_comp2, self.test_Tc_comp2 = [], [], [], []
            self.test_iter = []
            for iter, batch in enumerate(test_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2, _targets, _smiles1, _smiles2, Tb_comp1, Tc_comp1, Tb_comp2, Tc_comp2 = batch
                self.test_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.test_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.test_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.test_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.test_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.test_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.test_targets_list.append(_targets)
                self.test_smiles_list_comp1.append(_smiles1)
                self.test_smiles_list_comp2.append(_smiles2)
                self.test_Tb_comp1.append(Tb_comp1)
                self.test_Tc_comp1.append(Tc_comp1)
                self.test_Tb_comp2.append(Tb_comp2)
                self.test_Tc_comp2.append(Tc_comp2)
                self.test_iter.append(iter)

            self.all_batched_origin_graph_list_comp1, self.all_batched_origin_graph_list_comp2 = [], []
            self.all_batched_frag_graph_list_comp1, self.all_batched_frag_graph_list_comp2 = [], []
            self.all_batched_motif_graph_list_comp1, self.all_batched_motif_graph_list_comp2 = [], [] 
            self.all_targets_list = [] 
            self.all_smiles_list_comp1,self.all_smiles_list_comp2 = [], [] 
            self.all_Tb_comp1, self.all_Tc_comp1, self.all_Tb_comp2, self.all_Tc_comp2 = [], [], [], []
            self.all_iter = []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2, _targets, _smiles1, _smiles2, Tb_comp1, Tc_comp1, Tb_comp2, Tc_comp2 = batch
                self.all_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.all_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.all_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.all_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.all_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.all_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.all_targets_list.append(_targets)
                self.all_smiles_list_comp1.append(_smiles1)
                self.all_smiles_list_comp2.append(_smiles2)
                self.all_Tb_comp1.append(Tb_comp1)
                self.all_Tc_comp1.append(Tc_comp1)
                self.all_Tb_comp2.append(Tb_comp2)
                self.all_Tc_comp2.append(Tc_comp2)
                self.all_iter.append(iter)

        elif frag == 2:
            self.all_batched_origin_graph_list_comp1, self.all_batched_origin_graph_list_comp2 = [], []
            self.all_batched_frag_graph_list_comp1, self.all_batched_frag_graph_list_comp2 = [], []
            self.all_batched_motif_graph_list_comp1, self.all_batched_motif_graph_list_comp2 = [], [] 
            self.all_targets_list = [] 
            self.all_smiles_list_comp1,self.all_smiles_list_comp2 = [], [] 
            self.all_iter = []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2, _targets, _smiles1, _smiles2 = batch
                self.all_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.all_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.all_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.all_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.all_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.all_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.all_targets_list.append(_targets)
                self.all_smiles_list_comp1.append(_smiles1)
                self.all_smiles_list_comp2.append(_smiles2)
                self.all_iter.append(iter)
            
        elif frag == True:
            self.train_batched_origin_graph_list_comp1, self.train_batched_origin_graph_list_comp2 = [], []
            self.train_batched_frag_graph_list_comp1, self.train_batched_frag_graph_list_comp2 = [], []
            self.train_batched_motif_graph_list_comp1, self.train_batched_motif_graph_list_comp2 = [], [] 
            self.train_targets_list = [] 
            self.train_smiles_list_comp1,self.train_smiles_list_comp2 = [], [] 
            # self.train_names_list_comp1, self.train_names_list_comp1 = [], []
            self.train_iter = []
            for iter, batch in enumerate(train_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2, _targets, _smiles1,_smiles2 = batch
                self.train_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.train_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.train_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.train_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.train_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.train_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.train_targets_list.append(_targets)
                self.train_smiles_list_comp1.append(_smiles1)
                self.train_smiles_list_comp2.append(_smiles2)
                # self.train_names_list.append(_smiles)
                self.train_iter.append(iter)

            self.val_batched_origin_graph_list_comp1, self.val_batched_origin_graph_list_comp2 = [], []
            self.val_batched_frag_graph_list_comp1, self.val_batched_frag_graph_list_comp2 = [], []
            self.val_batched_motif_graph_list_comp1, self.val_batched_motif_graph_list_comp2 = [], [] 
            self.val_targets_list = [] 
            self.val_smiles_list_comp1,self.val_smiles_list_comp2 = [], [] 
            self.val_iter = []
            for iter, batch in enumerate(val_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2, _targets, _smiles1, _smiles2 = batch
                self.val_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.val_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.val_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.val_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.val_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.val_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.val_targets_list.append(_targets)
                self.val_smiles_list_comp1.append(_smiles1)
                self.val_smiles_list_comp2.append(_smiles2)
                self.val_iter.append(iter)

            self.test_batched_origin_graph_list_comp1, self.test_batched_origin_graph_list_comp2 = [], []
            self.test_batched_frag_graph_list_comp1, self.test_batched_frag_graph_list_comp2 = [], []
            self.test_batched_motif_graph_list_comp1, self.test_batched_motif_graph_list_comp2 = [], [] 
            self.test_targets_list = [] 
            self.test_smiles_list_comp1,self.test_smiles_list_comp2 = [], [] 
            self.test_iter = []
            for iter, batch in enumerate(test_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2, _targets, _smiles1, _smiles2 = batch
                self.test_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.test_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.test_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.test_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.test_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.test_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.test_targets_list.append(_targets)
                self.test_smiles_list_comp1.append(_smiles1)
                self.test_smiles_list_comp2.append(_smiles2)
                self.test_iter.append(iter)

            self.all_batched_origin_graph_list_comp1, self.all_batched_origin_graph_list_comp2 = [], []
            self.all_batched_frag_graph_list_comp1, self.all_batched_frag_graph_list_comp2 = [], []
            self.all_batched_motif_graph_list_comp1, self.all_batched_motif_graph_list_comp2 = [], [] 
            self.all_targets_list = [] 
            self.all_smiles_list_comp1,self.all_smiles_list_comp2 = [], [] 
            self.all_iter = []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph_comp1,_batched_origin_graph_comp2,_batched_frag_graph_comp1,_batched_frag_graph_comp2,_batched_motif_graph_comp1,_batched_motif_graph_comp2, _targets, _smiles1, _smiles2 = batch
                self.all_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.all_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.all_batched_frag_graph_list_comp1.append(_batched_frag_graph_comp1)
                self.all_batched_frag_graph_list_comp2.append(_batched_frag_graph_comp2)
                self.all_batched_motif_graph_list_comp1.append(_batched_motif_graph_comp1)
                self.all_batched_motif_graph_list_comp2.append(_batched_motif_graph_comp2)
                self.all_targets_list.append(_targets)
                self.all_smiles_list_comp1.append(_smiles1)
                self.all_smiles_list_comp2.append(_smiles2)
                self.all_iter.append(iter)

        else:
            self.train_batched_origin_graph_list_comp1, self.train_batched_origin_graph_list_comp2 = [], [] 
            self.train_targets_list= []
            self.train_smiles_list_comp1, self.train_smiles_list_comp2 = [], []
            # self.train_names_list = []
            self.train_iter = []
            for iter, batch in enumerate(train_loader):
                _batched_origin_graph_comp1, _batched_origin_graph_comp2, _targets, _smiles1, _smiles2 = batch
                self.train_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.train_batched_frag_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.train_targets_list.append(_targets)
                self.train_smiles_list_comp1.append(_smiles1)
                self.train_smiles_list_comp2.append(_smiles2)
                # self.train_names_list.append(_smiles2)
                self.train_iter.append(iter)

            self.val_batched_origin_graph_list_comp1, self.val_batched_origin_graph_list_comp2 = [], [] 
            self.val_targets_list= []
            self.val_smiles_list_comp1, self.val_smiles_list_comp2 = [], []
            self.val_iter = []
            for iter, batch in enumerate(val_loader):
                _batched_origin_graph_comp1, _batched_origin_graph_comp1, _targets, _smiles1, _smiles2 = batch
                self.val_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.val_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.val_targets_list.append(_targets)
                self.val_smiles_list_comp1.append(_smiles1)
                self.val_smiles_list_comp2.append(_smiles2)
                self.val_iter.append(iter)

            self.test_batched_origin_graph_list_comp1, self.test_batched_origin_graph_list_comp2 = [], [] 
            self.test_targets_list= []
            self.test_smiles_list_comp1, self.test_smiles_list_comp2 = [], []
            self.test_iter = []
            for iter, batch in enumerate(test_loader):
                _batched_origin_graph_comp1, _batched_origin_graph_comp1, _targets, _smiles1, _smiles2 = batch
                self.test_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.test_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.test_targets_list.append(_targets)
                self.test_smiles_list_comp1.append(_smiles1)
                self.test_smiles_list_comp2.append(_smiles2)
                self.test_iter.append(iter)

            self.all_batched_origin_graph_list_comp1, self.all_batched_origin_graph_list_comp2 = [], [] 
            self.all_targets_list= []
            self.all_smiles_list_comp1, self.all_smiles_list_comp2 = [], []
            self.all_iter = []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph_comp1, _batched_origin_graph_comp1, _targets, _smiles1, _smiles2 = batch
                self.all_batched_origin_graph_list_comp1.append(_batched_origin_graph_comp1)
                self.all_batched_origin_graph_list_comp2.append(_batched_origin_graph_comp2)
                self.all_targets_list.append(_targets)
                self.all_smiles_list_comp1.append(_smiles1)
                self.all_smiles_list_comp2.append(_smiles2)
                self.all_iter.append(iter)



def train_epoch_frag_azeotrope(model, optimizer, iter, 
                               batched_origin_graph_comp1, batched_origin_graph_comp2, 
                               batched_frag_graph_comp1, batched_frag_graph_comp2, 
                               batched_motif_graph_comp1, batched_motif_graph_comp2,
                               targets):
    st = time.time()
    model.train()
    score_list = []
    target_list = []
    epoch_loss = 0
    output_list = []
    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device = 'cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device = 'cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)

        score = model.forward(batch_origin_graph_comp1, batch_origin_graph_comp2,
                                  batch_origin_node_comp1, batch_origin_node_comp2,
                                  batch_origin_edge_comp1, batch_origin_edge_comp2,
                                  batch_frag_graph_comp1, batch_frag_graph_comp2,
                                  batch_frag_node_comp1, batch_frag_node_comp2,
                                  batch_frag_edge_comp1, batch_frag_edge_comp2,
                                  batch_motif_graph_comp1, batch_motif_graph_comp2,
                                  batch_motif_node_comp1, batch_motif_node_comp2,
                                  batch_motif_edge_comp1, batch_motif_edge_comp2)
        target = targets[i].float().to(device='cpu')
        loss = model.loss(score,target)
        loss.backward()
        optimizer.step()
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()

    score_list = torch.cat(score_list, dim = 0)    
    target_list = torch.cat(target_list, dim = 0).int()
    output_list = score_list.tolist()
    for i in range(len(output_list)):
        target_list
        if output_list[i][0] >=0.5:
            output_list[i][0] = 1
        else:
            output_list[i][0] = 0
    output_list = torch.tensor(output_list)
    epoch_train_metrics = Metrics_classification(target_list.detach().to(device = 'cpu'),output_list.detach().to(device = 'cpu'))
        
    return model, epoch_loss, epoch_train_metrics
    
def evaluate_frag_azeotrope(model, iter, 
                            batched_origin_graph_comp1, batched_origin_graph_comp2, 
                            batched_frag_graph_comp1, batched_frag_graph_comp2,
                            batched_motif_graph_comp1, batched_motif_graph_comp2,
                            targets):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    output_list = []

    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device='cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device='cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device='cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device='cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device='cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device='cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')

        torch.autograd.set_detect_anomaly(False)

        score = model.forward(batch_origin_graph_comp1, batch_origin_graph_comp2,
                                  batch_origin_node_comp1, batch_origin_node_comp2,
                                  batch_origin_edge_comp1, batch_origin_edge_comp2,
                                  batch_frag_graph_comp1, batch_frag_graph_comp2,
                                  batch_frag_node_comp1, batch_frag_node_comp2,
                                  batch_frag_edge_comp1, batch_frag_edge_comp2,
                                  batch_motif_graph_comp1, batch_motif_graph_comp2,
                                  batch_motif_node_comp1, batch_motif_node_comp2,
                                  batch_motif_edge_comp1, batch_motif_edge_comp2)
        
        target = targets[i].float().to(device='cpu')
        loss = model.loss(score, target)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()

    score_list = torch.cat(score_list, dim = 0)    
    target_list = torch.cat(target_list, dim = 0).int()
    output_list = score_list.tolist()
    for i in range(len(output_list)):
        target_list
        if output_list[i][0] >=0.5:
            output_list[i][0] = 1
        else:
            output_list[i][0] = 0
    output_list = torch.tensor(output_list)
    epoch_eval_metrics = Metrics_classification(target_list.detach().to(device='cpu'),output_list.detach().to(device='cpu'))

    return epoch_loss, epoch_eval_metrics


def predict_Tb_Tc(model_Tb, model_Tc, iter, batched_origin_graph_comp1, batched_origin_graph_comp2, 
                            batched_frag_graph_comp1, batched_frag_graph_comp2,
                            batched_motif_graph_comp1, batched_motif_graph_comp2):
    model_Tb.eval()
    Tb1 = []
    Tc1 = []
    Tb2 = []
    Tc2 = []

    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device='cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device='cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device='cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device='cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device='cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device='cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')

        torch.autograd.set_detect_anomaly(False)

        score_Tb = model_Tb.forward(batch_origin_graph_comp1,
                                  batch_origin_node_comp1,
                                  batch_origin_edge_comp1,
                                  batch_frag_graph_comp1,
                                  batch_frag_node_comp1,
                                  batch_frag_edge_comp1,
                                  batch_motif_graph_comp1,
                                  batch_motif_node_comp1,
                                  batch_motif_edge_comp1)
        
        Tb1.append(score_Tb)

        score_Tc = model_Tc.forward(batch_origin_graph_comp1,
                                  batch_origin_node_comp1,
                                  batch_origin_edge_comp1,
                                  batch_frag_graph_comp1,
                                  batch_frag_node_comp1,
                                  batch_frag_edge_comp1,
                                  batch_motif_graph_comp1,
                                  batch_motif_node_comp1,
                                  batch_motif_edge_comp1)
        Tc1.append(score_Tc)

        score_Tb = model_Tb.forward(batch_origin_graph_comp2,
                                  batch_origin_node_comp2,
                                  batch_origin_edge_comp2,
                                  batch_frag_graph_comp2,
                                  batch_frag_node_comp2,
                                  batch_frag_edge_comp2,
                                  batch_motif_graph_comp2,
                                  batch_motif_node_comp2,
                                  batch_motif_edge_comp2)
        Tb2.append(score_Tb)

        score_Tc = model_Tc.forward(batch_origin_graph_comp2,
                                  batch_origin_node_comp2,
                                  batch_origin_edge_comp2,
                                  batch_frag_graph_comp2,
                                  batch_frag_node_comp2,
                                  batch_frag_edge_comp2,
                                  batch_motif_graph_comp2,
                                  batch_motif_node_comp2,
                                  batch_motif_edge_comp2)
        Tc2.append(score_Tc)



    Tb1 = torch.cat(Tb1, dim = 0)
    comp1_Tb = Tb1.tolist()

    Tb2 = torch.cat(Tb2, dim = 0)
    comp2_Tb = Tb2.tolist()

    Tc1 = torch.cat(Tc1, dim = 0)
    comp1_Tc = Tc1.tolist()

    Tc2 = torch.cat(Tc2, dim = 0)
    comp2_Tc = Tc2.tolist()
    return comp1_Tb, comp2_Tb, comp1_Tc, comp2_Tc


def train_epoch_frag_clf(model, optimizer, iter, 
                               batched_origin_graph_comp1, batched_origin_graph_comp2, 
                               batched_frag_graph_comp1, batched_frag_graph_comp2, 
                               batched_motif_graph_comp1, batched_motif_graph_comp2,
                               batched_Tb_comp1, batched_Tc_comp1, batched_Tb_comp2, batched_Tc_comp2,
                               targets):
    st = time.time()
    model.train()
    score_list = []
    target_list = []
    epoch_loss = 0
    num_batch = 0
    output_list = []
    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device = 'cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device = 'cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')
        batch_Tb_comp1 = batched_Tb_comp1[i].to(device = 'cpu')
        batch_Tc_comp1 = batched_Tc_comp1[i].to(device = 'cpu')
        batch_Tb_comp2 = batched_Tb_comp2[i].to(device = 'cpu')
        batch_Tc_comp2 = batched_Tc_comp2[i].to(device = 'cpu')

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)

        score = model.forward(batch_origin_graph_comp1, batch_origin_graph_comp2,
                                  batch_origin_node_comp1, batch_origin_node_comp2,
                                  batch_origin_edge_comp1, batch_origin_edge_comp2,
                                  batch_frag_graph_comp1, batch_frag_graph_comp2,
                                  batch_frag_node_comp1, batch_frag_node_comp2,
                                  batch_frag_edge_comp1, batch_frag_edge_comp2,
                                  batch_motif_graph_comp1, batch_motif_graph_comp2,
                                  batch_motif_node_comp1, batch_motif_node_comp2,
                                  batch_motif_edge_comp1, batch_motif_edge_comp2,
                                  batch_Tb_comp1, batch_Tc_comp1,
                                  batch_Tb_comp2, batch_Tc_comp2)
        target = targets[i].float().to(device='cpu')
        loss = model.loss(score,target)
        loss.backward()
        optimizer.step()
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
        num_batch += 1
    
    target_list = torch.cat(target_list, dim = 0).int()
    for i in range(len(score_list[0])):
        if score_list[0][i][0] >=0.5:
            output_list.append([1])
        else:
            output_list.append([0])
    output_list = torch.tensor(output_list)

    epoch_train_metrics = Metrics_classification(target_list.detach().to(device = 'cpu'), output_list.detach().to(device = 'cpu'))

    return model, epoch_loss/num_batch, epoch_train_metrics

        
def evaluate_frag_clf(model, iter, 
                               batched_origin_graph_comp1, batched_origin_graph_comp2, 
                               batched_frag_graph_comp1, batched_frag_graph_comp2, 
                               batched_motif_graph_comp1, batched_motif_graph_comp2,
                               batched_Tb_comp1, batched_Tc_comp1, batched_Tb_comp2, batched_Tc_comp2,
                               targets):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    num_batch = 0
    output_list = []

    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device='cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device='cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device='cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device='cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device='cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device='cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')
        batch_Tb_comp1 = batched_Tb_comp1[i].to(device = 'cpu')
        batch_Tc_comp1 = batched_Tc_comp1[i].to(device = 'cpu')
        batch_Tb_comp2 = batched_Tb_comp2[i].to(device = 'cpu')
        batch_Tc_comp2 = batched_Tc_comp2[i].to(device = 'cpu')

        torch.autograd.set_detect_anomaly(False)

        score = model.forward(batch_origin_graph_comp1, batch_origin_graph_comp2,
                                  batch_origin_node_comp1, batch_origin_node_comp2,
                                  batch_origin_edge_comp1, batch_origin_edge_comp2,
                                  batch_frag_graph_comp1, batch_frag_graph_comp2,
                                  batch_frag_node_comp1, batch_frag_node_comp2,
                                  batch_frag_edge_comp1, batch_frag_edge_comp2,
                                  batch_motif_graph_comp1, batch_motif_graph_comp2,
                                  batch_motif_node_comp1, batch_motif_node_comp2,
                                  batch_motif_edge_comp1, batch_motif_edge_comp2,
                                  batch_Tb_comp1, batch_Tc_comp1,
                                  batch_Tb_comp2, batch_Tc_comp2)
        
        target = targets[i].float().to(device='cpu')
        loss = model.loss(score, target)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
        num_batch += 1

    target_list = torch.cat(target_list, dim = 0).int()
    for i in range(len(score_list[0])):
        if score_list[0][i][0] >=0.5:
            output_list.append([1])
        else:
            output_list.append([0])
    output_list = torch.tensor(output_list)
    epoch_eval_metrics = Metrics_classification(target_list.detach().to(device='cpu'), output_list.detach().to(device='cpu'))

    return epoch_loss/num_batch, epoch_eval_metrics

def evaluate_frag_clf_attention_value(model, iter, 
                                        batched_origin_graph_comp1, batched_origin_graph_comp2, 
                                        batched_frag_graph_comp1, batched_frag_graph_comp2, 
                                        batched_motif_graph_comp1, batched_motif_graph_comp2,
                                        # batched_Tb_comp1, batched_Tc_comp1, batched_Tb_comp2, batched_Tc_comp2,
                                        targets,smiles_list_comp1,smiles_list_comp2):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    num_batch = 0
    attention_comp1_list = []
    attention_comp2_list = []
    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device='cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device='cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device='cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device='cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device='cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device='cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')
        # batch_Tb_comp1 = batched_Tb_comp1[i].to(device = 'cpu')
        # batch_Tc_comp1 = batched_Tc_comp1[i].to(device = 'cpu')
        # batch_Tb_comp2 = batched_Tb_comp2[i].to(device = 'cpu')
        # batch_Tc_comp2 = batched_Tc_comp2[i].to(device = 'cpu')

        torch.autograd.set_detect_anomaly(False)

        score, attention_comp1, attention_comp2 = model.forward(batch_origin_graph_comp1, batch_origin_graph_comp2,
                                  batch_origin_node_comp1, batch_origin_node_comp2,
                                  batch_origin_edge_comp1, batch_origin_edge_comp2,
                                  batch_frag_graph_comp1, batch_frag_graph_comp2,
                                  batch_frag_node_comp1, batch_frag_node_comp2,
                                  batch_frag_edge_comp1, batch_frag_edge_comp2,
                                  batch_motif_graph_comp1, batch_motif_graph_comp2,
                                  batch_motif_node_comp1, batch_motif_node_comp2,
                                  batch_motif_edge_comp1, batch_motif_edge_comp2,
                                #   batch_Tb_comp1, batch_Tc_comp1,
                                #   batch_Tb_comp2, batch_Tc_comp2,
                                  get_descriptors=False, get_attention=True)
        
        target = targets[i].float().to(device='cpu')
        score_list.append(score)
        target_list.append(target)
        attention_comp1_list.extend(attention_comp1)
        attention_comp2_list.extend(attention_comp2)
 
    target_list = torch.cat(target_list, dim = 0).int()
    for i in range(len(score_list)):
        if score_list[i][0] >=0.5:
            score_list[i][0] = 1
        else:
            score_list[i][0] = 0
    score_list = torch.cat(score_list, dim = 0).int()
    epoch_eval_metrics = Metrics_classification(target_list.detach().to(device='cpu'),score_list.detach().to(device='cpu'))

    predict = score_list.detach().to(device='cpu')
    true = target_list.detach().to(device='cpu')
    
    return epoch_loss, epoch_eval_metrics, predict, true, attention_comp1_list, attention_comp2_list, smiles_list_comp1, smiles_list_comp2


def evaluate_frag_clf_attention_value2(model, iter, 
                                        batched_origin_graph_comp1, batched_origin_graph_comp2, 
                                        batched_frag_graph_comp1, batched_frag_graph_comp2, 
                                        batched_motif_graph_comp1, batched_motif_graph_comp2,
                                        batched_Tb_comp1, batched_Tc_comp1, batched_Tb_comp2, batched_Tc_comp2,
                                        targets,smiles_list_comp1,smiles_list_comp2):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    output_list = []
    attention_comp1_list = []
    attention_comp2_list = []
    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device='cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device='cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device='cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device='cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device='cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device='cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')
        batch_Tb_comp1 = batched_Tb_comp1[i].to(device = 'cpu')
        batch_Tc_comp1 = batched_Tc_comp1[i].to(device = 'cpu')
        batch_Tb_comp2 = batched_Tb_comp2[i].to(device = 'cpu')
        batch_Tc_comp2 = batched_Tc_comp2[i].to(device = 'cpu')

        torch.autograd.set_detect_anomaly(False)

        score, attention_comp1, attention_comp2 = model.forward(batch_origin_graph_comp1, batch_origin_graph_comp2,
                                  batch_origin_node_comp1, batch_origin_node_comp2,
                                  batch_origin_edge_comp1, batch_origin_edge_comp2,
                                  batch_frag_graph_comp1, batch_frag_graph_comp2,
                                  batch_frag_node_comp1, batch_frag_node_comp2,
                                  batch_frag_edge_comp1, batch_frag_edge_comp2,
                                  batch_motif_graph_comp1, batch_motif_graph_comp2,
                                  batch_motif_node_comp1, batch_motif_node_comp2,
                                  batch_motif_edge_comp1, batch_motif_edge_comp2,
                                  batch_Tb_comp1, batch_Tc_comp1,
                                  batch_Tb_comp2, batch_Tc_comp2,
                                  get_descriptors=False, get_attention=True)
        
        target = targets[i].float().to(device='cpu')
        score_list.append(score)
        target_list.append(target)
        attention_comp1_list.extend(attention_comp1)
        attention_comp2_list.extend(attention_comp2)

    score_list = torch.cat(score_list, dim = 0)    
    target_list = torch.cat(target_list, dim = 0).int()
    output_list = score_list.tolist()
    for i in range(len(output_list)):
        target_list
        if output_list[i][0] >=0.5:
            output_list[i][0] = 1
        else:
            output_list[i][0] = 0
    output_list = torch.tensor(output_list)
    # output_list = torch.tensor(score_list)
    epoch_eval_metrics = Metrics_classification(target_list.detach().to(device='cpu'),output_list.detach().to(device='cpu'))

    predict = output_list.detach().to(device='cpu')
    true = target_list.detach().to(device='cpu')
    
    return epoch_loss, epoch_eval_metrics, predict, true, attention_comp1_list, attention_comp2_list, smiles_list_comp1, smiles_list_comp2, score_list


def train_epoch_frag_VLE(model, optimizer, scaling, iter, 
                               batched_origin_graph_comp1, batched_origin_graph_comp2, 
                               batched_frag_graph_comp1, batched_frag_graph_comp2, 
                               batched_motif_graph_comp1, batched_motif_graph_comp2,
                               batched_Tb_comp1, batched_Tc_comp1, batched_Tb_comp2, batched_Tc_comp2,
                               batched_x1, targets, n_param = None):
    st = time.time()
    model.train()
    score_list = []
    target_list = []
    epoch_loss = 0

    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device = 'cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device = 'cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')
        batch_Tb_comp1 = batched_Tb_comp1[i].to(device = 'cpu')
        batch_Tc_comp1 = batched_Tc_comp1[i].to(device = 'cpu')
        batch_Tb_comp2 = batched_Tb_comp2[i].to(device = 'cpu')
        batch_Tc_comp2 = batched_Tc_comp2[i].to(device = 'cpu')
        batch_x1 = batched_x1[i].to(device = 'cpu')

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)

        score = model.forward(batch_origin_graph_comp1, batch_origin_graph_comp2,
                                  batch_origin_node_comp1, batch_origin_node_comp2,
                                  batch_origin_edge_comp1, batch_origin_edge_comp2,
                                  batch_frag_graph_comp1, batch_frag_graph_comp2,
                                  batch_frag_node_comp1, batch_frag_node_comp2,
                                  batch_frag_edge_comp1, batch_frag_edge_comp2,
                                  batch_motif_graph_comp1, batch_motif_graph_comp2,
                                  batch_motif_node_comp1, batch_motif_node_comp2,
                                  batch_motif_edge_comp1, batch_motif_edge_comp2,
                                  batch_Tb_comp1, batch_Tc_comp1,
                                  batch_Tb_comp2, batch_Tc_comp2,
                                  batch_x1)
        target = targets[i].float().to(device='cpu')
        loss = model.loss(score,target)
        loss.backward()
        optimizer.step()
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim = 0)    
    target_list = torch.cat(target_list, dim = 0)
    epoch_train_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device = 'cpu')), 
                                scaling.ReScaler(score_list.detach().to(device = 'cpu')),n_param)

    return model, epoch_loss, epoch_train_metrics

def evaluate_frag_VLE(model, scaling, iter, 
                               batched_origin_graph_comp1, batched_origin_graph_comp2, 
                               batched_frag_graph_comp1, batched_frag_graph_comp2, 
                               batched_motif_graph_comp1, batched_motif_graph_comp2,
                               batched_Tb_comp1, batched_Tc_comp1, batched_Tb_comp2, batched_Tc_comp2,
                               batched_x1, targets, n_param = None):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device = 'cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device = 'cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')
        batch_Tb_comp1 = batched_Tb_comp1[i].to(device = 'cpu')
        batch_Tc_comp1 = batched_Tc_comp1[i].to(device = 'cpu')
        batch_Tb_comp2 = batched_Tb_comp2[i].to(device = 'cpu')
        batch_Tc_comp2 = batched_Tc_comp2[i].to(device = 'cpu')
        batch_x1 = batched_x1[i].to(device = 'cpu')
        torch.autograd.set_detect_anomaly(False)

        score = model.forward(batch_origin_graph_comp1, batch_origin_graph_comp2,
                                  batch_origin_node_comp1, batch_origin_node_comp2,
                                  batch_origin_edge_comp1, batch_origin_edge_comp2,
                                  batch_frag_graph_comp1, batch_frag_graph_comp2,
                                  batch_frag_node_comp1, batch_frag_node_comp2,
                                  batch_frag_edge_comp1, batch_frag_edge_comp2,
                                  batch_motif_graph_comp1, batch_motif_graph_comp2,
                                  batch_motif_node_comp1, batch_motif_node_comp2,
                                  batch_motif_edge_comp1, batch_motif_edge_comp2,
                                  batch_Tb_comp1, batch_Tc_comp1,
                                  batch_Tb_comp2, batch_Tc_comp2,
                                  batch_x1)
        target = targets[i].float().to(device='cpu')
        loss = model.loss(score, target)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    epoch_eval_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)

    predict = scaling.ReScaler(score_list.detach().to(device='cpu'))
    true = scaling.ReScaler(target_list.detach().to(device='cpu'))

    return epoch_loss, epoch_eval_metrics


def train_epoch_frag_VLE4(model, optimizer, scaling, iter, 
                               batched_origin_graph_comp1, batched_origin_graph_comp2, 
                               batched_frag_graph_comp1, batched_frag_graph_comp2, 
                               batched_motif_graph_comp1, batched_motif_graph_comp2,
                               batched_Tb_comp1, batched_Tc_comp1, batched_Tb_comp2, batched_Tc_comp2,
                               batched_x1, targets, n_param = None,lam = 0.1):
    st = time.time()
    model.train()
    score_list = []
    target_list = []
    epoch_loss = 0

    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device = 'cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device = 'cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')
        batch_Tb_comp1 = batched_Tb_comp1[i].to(device = 'cpu')
        batch_Tc_comp1 = batched_Tc_comp1[i].to(device = 'cpu')
        batch_Tb_comp2 = batched_Tb_comp2[i].to(device = 'cpu')
        batch_Tc_comp2 = batched_Tc_comp2[i].to(device = 'cpu')
        batch_x1 = batched_x1[i].to(device = 'cpu')

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)

        score, var = model.forward(batch_origin_graph_comp1, batch_origin_graph_comp2,
                                  batch_origin_node_comp1, batch_origin_node_comp2,
                                  batch_origin_edge_comp1, batch_origin_edge_comp2,
                                  batch_frag_graph_comp1, batch_frag_graph_comp2,
                                  batch_frag_node_comp1, batch_frag_node_comp2,
                                  batch_frag_edge_comp1, batch_frag_edge_comp2,
                                  batch_motif_graph_comp1, batch_motif_graph_comp2,
                                  batch_motif_node_comp1, batch_motif_node_comp2,
                                  batch_motif_edge_comp1, batch_motif_edge_comp2,
                                  batch_Tb_comp1, batch_Tc_comp1,
                                  batch_Tb_comp2, batch_Tc_comp2,
                                  batch_x1)
        target = targets[i].float().to(device='cpu')
        loss = model.loss(score,target,var,lam)
        loss.backward()
        optimizer.step()
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim = 0)    
    target_list = torch.cat(target_list, dim = 0)
    epoch_train_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device = 'cpu')), 
                                scaling.ReScaler(score_list.detach().to(device = 'cpu')),n_param)

    return model, epoch_loss, epoch_train_metrics


def evaluate_frag_VLE4(model, scaling, iter, 
                               batched_origin_graph_comp1, batched_origin_graph_comp2, 
                               batched_frag_graph_comp1, batched_frag_graph_comp2, 
                               batched_motif_graph_comp1, batched_motif_graph_comp2,
                               batched_Tb_comp1, batched_Tc_comp1, batched_Tb_comp2, batched_Tc_comp2,
                               batched_x1, targets, smiles_comp1, smiles_comp2,n_param = None,lam = 0.1, flag = False):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device = 'cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device = 'cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')
        batch_Tb_comp1 = batched_Tb_comp1[i].to(device = 'cpu')
        batch_Tc_comp1 = batched_Tc_comp1[i].to(device = 'cpu')
        batch_Tb_comp2 = batched_Tb_comp2[i].to(device = 'cpu')
        batch_Tc_comp2 = batched_Tc_comp2[i].to(device = 'cpu')
        batch_x1 = batched_x1[i].to(device = 'cpu')
        torch.autograd.set_detect_anomaly(False)

        score,var = model.forward(batch_origin_graph_comp1, batch_origin_graph_comp2,
                                  batch_origin_node_comp1, batch_origin_node_comp2,
                                  batch_origin_edge_comp1, batch_origin_edge_comp2,
                                  batch_frag_graph_comp1, batch_frag_graph_comp2,
                                  batch_frag_node_comp1, batch_frag_node_comp2,
                                  batch_frag_edge_comp1, batch_frag_edge_comp2,
                                  batch_motif_graph_comp1, batch_motif_graph_comp2,
                                  batch_motif_node_comp1, batch_motif_node_comp2,
                                  batch_motif_edge_comp1, batch_motif_edge_comp2,
                                  batch_Tb_comp1, batch_Tc_comp1,
                                  batch_Tb_comp2, batch_Tc_comp2,
                                  batch_x1)
        target = targets[i].float().to(device='cpu')
        loss = model.loss(score, target, var, lam)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    epoch_eval_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)

    predict = scaling.ReScaler(score_list.detach().to(device='cpu'))
    true = scaling.ReScaler(target_list.detach().to(device='cpu'))

    if flag:
        return epoch_loss, epoch_eval_metrics, predict, true, smiles_comp1, smiles_comp2
    else:
        return epoch_loss, epoch_eval_metrics


def evaluate_frag_VLE_attention_value(model, scaling, iter, 
                               batched_origin_graph_comp1, batched_origin_graph_comp2, 
                               batched_frag_graph_comp1, batched_frag_graph_comp2, 
                               batched_motif_graph_comp1, batched_motif_graph_comp2,
                               batched_Tb_comp1, batched_Tc_comp1, batched_Tb_comp2, batched_Tc_comp2,
                               batched_x1, targets,smiles_list_comp1, smiles_list_comp2, n_param = None):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    attention_comp1_list = []
    attention_comp2_list = []
    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device='cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device='cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device='cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device='cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device='cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device='cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')
        batch_Tb_comp1 = batched_Tb_comp1[i].to(device = 'cpu')
        batch_Tc_comp1 = batched_Tc_comp1[i].to(device = 'cpu')
        batch_Tb_comp2 = batched_Tb_comp2[i].to(device = 'cpu')
        batch_Tc_comp2 = batched_Tc_comp2[i].to(device = 'cpu')
        batch_x1 = batched_x1[i].to(device = 'cpu')
        torch.autograd.set_detect_anomaly(False)

        score, attention_comp1, attention_comp2 = model.forward(batch_origin_graph_comp1, batch_origin_graph_comp2,
                                  batch_origin_node_comp1, batch_origin_node_comp2,
                                  batch_origin_edge_comp1, batch_origin_edge_comp2,
                                  batch_frag_graph_comp1, batch_frag_graph_comp2,
                                  batch_frag_node_comp1, batch_frag_node_comp2,
                                  batch_frag_edge_comp1, batch_frag_edge_comp2,
                                  batch_motif_graph_comp1, batch_motif_graph_comp2,
                                  batch_motif_node_comp1, batch_motif_node_comp2,
                                  batch_motif_edge_comp1, batch_motif_edge_comp2,
                                  batch_Tb_comp1, batch_Tc_comp1,
                                  batch_Tb_comp2, batch_Tc_comp2,
                                  batch_x1,
                                  get_descriptors=False, get_attention=True)
        attention_comp1_list.extend(attention_comp1)
        attention_comp2_list.extend(attention_comp2)
        target = targets[i].float().to(device='cpu')
        loss = model.loss(score, target)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    epoch_eval_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')),n_param)

    predict = scaling.ReScaler(score_list.detach().to(device='cpu'))
    true = scaling.ReScaler(target_list.detach().to(device='cpu'))

    
    return epoch_loss, epoch_eval_metrics, predict, true, attention_comp1_list, attention_comp2_list,smiles_list_comp1, smiles_list_comp2


def evaluate_frag_VLE_attention_value4(model, scaling, iter, 
                               batched_origin_graph_comp1, batched_origin_graph_comp2, 
                               batched_frag_graph_comp1, batched_frag_graph_comp2, 
                               batched_motif_graph_comp1, batched_motif_graph_comp2,
                               batched_Tb_comp1, batched_Tc_comp1, batched_Tb_comp2, batched_Tc_comp2,
                               batched_x1, targets,smiles_list_comp1, smiles_list_comp2, n_param = None, lam = 0.5):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    attention_comp1_list = []
    attention_comp2_list = []
    for i in iter:
        batch_origin_node_comp1 = batched_origin_graph_comp1[i].ndata['feat'].to(device='cpu')
        batch_origin_node_comp2 = batched_origin_graph_comp2[i].ndata['feat'].to(device='cpu')
        batch_origin_edge_comp1 = batched_origin_graph_comp1[i].edata['feat'].to(device='cpu')
        batch_origin_edge_comp2 = batched_origin_graph_comp2[i].edata['feat'].to(device='cpu')
        batch_origin_graph_comp1 = batched_origin_graph_comp1[i].to(device='cpu')
        batch_origin_graph_comp2 = batched_origin_graph_comp2[i].to(device='cpu')

        batch_frag_node_comp1 = batched_frag_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_frag_node_comp2 = batched_frag_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_frag_edge_comp1 = batched_frag_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_frag_edge_comp2 = batched_frag_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_frag_graph_comp1 = batched_frag_graph_comp1[i].to(device = 'cpu')
        batch_frag_graph_comp2 = batched_frag_graph_comp2[i].to(device = 'cpu')

        batch_motif_node_comp1 = batched_motif_graph_comp1[i].ndata['feat'].to(device = 'cpu')
        batch_motif_node_comp2 = batched_motif_graph_comp2[i].ndata['feat'].to(device = 'cpu')
        batch_motif_edge_comp1 = batched_motif_graph_comp1[i].edata['feat'].to(device = 'cpu')
        batch_motif_edge_comp2 = batched_motif_graph_comp2[i].edata['feat'].to(device = 'cpu')
        batch_motif_graph_comp1 = batched_motif_graph_comp1[i].to(device = 'cpu')
        batch_motif_graph_comp2 = batched_motif_graph_comp2[i].to(device = 'cpu')
        batch_Tb_comp1 = batched_Tb_comp1[i].to(device = 'cpu')
        batch_Tc_comp1 = batched_Tc_comp1[i].to(device = 'cpu')
        batch_Tb_comp2 = batched_Tb_comp2[i].to(device = 'cpu')
        batch_Tc_comp2 = batched_Tc_comp2[i].to(device = 'cpu')
        batch_x1 = batched_x1[i].to(device = 'cpu')
        torch.autograd.set_detect_anomaly(False)

        score, var, attention_comp1, attention_comp2 = model.forward(batch_origin_graph_comp1, batch_origin_graph_comp2,
                                  batch_origin_node_comp1, batch_origin_node_comp2,
                                  batch_origin_edge_comp1, batch_origin_edge_comp2,
                                  batch_frag_graph_comp1, batch_frag_graph_comp2,
                                  batch_frag_node_comp1, batch_frag_node_comp2,
                                  batch_frag_edge_comp1, batch_frag_edge_comp2,
                                  batch_motif_graph_comp1, batch_motif_graph_comp2,
                                  batch_motif_node_comp1, batch_motif_node_comp2,
                                  batch_motif_edge_comp1, batch_motif_edge_comp2,
                                  batch_Tb_comp1, batch_Tc_comp1,
                                  batch_Tb_comp2, batch_Tc_comp2,
                                  batch_x1,
                                  get_descriptors=False, get_attention=True)
        attention_comp1_list.extend(attention_comp1)
        attention_comp2_list.extend(attention_comp2)
        target = targets[i].float().to(device='cpu')
        loss = model.loss(score, target, var, lam)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    epoch_eval_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')),n_param)

    predict = scaling.ReScaler(score_list.detach().to(device='cpu'))
    true = scaling.ReScaler(target_list.detach().to(device='cpu'))

    
    return epoch_loss, epoch_eval_metrics, predict, true, attention_comp1_list, attention_comp2_list,smiles_list_comp1, smiles_list_comp2