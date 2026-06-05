
import os
import dgl.backend as F
import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import save_graphs, load_graphs
from utils.mol2graph import graph_2_frag, create_channels

class pair_Dataset(object):

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

    def _prepare(self, params, smiles_2_graph, atom_featurizer, bond_featurizer, mol_featurizer, load, log_every, error_log):
        '''
        :param
        '''
        cache_comp1_path = os.path.join(self.cache_file_path, params['Dataset'] + '_comp1_' + self.name + '_CCC')
        cache_comp2_path = os.path.join(self.cache_file_path, params['Dataset'] + '_comp2_' + self.name +'_CCC')

        if os.path.exists(cache_comp1_path) and os.path.exists(cache_comp2_path) and load:
            print('Loading saved dgl graphs ...')
            self.origin_graphs_comp1, label_dict_comp1, = load_graphs(cache_comp1_path)
            self.origin_graphs_comp2, label_dict_comp2 = load_graphs(cache_comp2_path)
            self.values = label_dict_comp1['values']
            self.Tb_comp1 = label_dict_comp1['Tb_comp1']
            self.Tc_comp1 = label_dict_comp1['Tc_comp1']
            self.Tb_comp2 = label_dict_comp2['Tb_comp2']
            self.Tc_comp2 = label_dict_comp2['Tc_comp2']
            valid_idx = label_dict_comp1['valid_idx']
            self.valid_idx = valid_idx.detach().numpy().tolist()
        else:
            print('Preparing dgl by featurizers ...')
            self.origin_graphs_comp1 = []
            for i, s in enumerate(self.df['comp1']):
                if (i + 1) % log_every == 0:
                    print('Currently preparing molecule {:d}/{:d} for comp1'.format(i + 1, len(self)))
                self.origin_graphs_comp1.append(smiles_2_graph(s, atom_featurizer, bond_featurizer, mol_featurizer))

            self.origin_graphs_comp2 = []
            for i, s in enumerate(self.df['comp2']):
                if (i + 1) % log_every == 0:
                    print('Currently preparing molecule {:d}/{:d} for comp2'.format(i + 1, len(self)))
                self.origin_graphs_comp2.append(smiles_2_graph(s, atom_featurizer, bond_featurizer, mol_featurizer))

            self.valid_idx = []
            origin_graphs_comp1 = []
            origin_graphs_comp2 = []
            failed_smiles_comp1 = []
            failed_smiles_comp2 = []
            for i in range(len(self.origin_graphs_comp1)):
                if self.origin_graphs_comp1[i] is not None:
                    if self.origin_graphs_comp2[i] is not None:
                        self.valid_idx.append(i)
                        origin_graphs_comp1.append(self.origin_graphs_comp1[i])
                        origin_graphs_comp2.append(self.origin_graphs_comp2[i])
                    else:
                        failed_smiles_comp2.append(self.df['comp2'][i])
                else:
                    failed_smiles_comp1.append(self.df['comp1'][i])
            if error_log is not None:
                df = open(error_log,'w')
                if len(failed_smiles_comp1) > 0:
                    df.write('index'+','+'comp1'+'\n')
                    for i in len(range(failed_smiles_comp1)):
                        df.write('"{}","{}"'.format(str(i+1),failed_smiles_comp1[i]))
                        df.write('\n')
                if len(failed_smiles_comp2) > 0:
                    df.write('index'+','+'comp2'+'\n')
                    for i in len(range(failed_smiles_comp1)):
                        df.write('"{}","{}"'.format(str(i+1),failed_smiles_comp1[i]))
                        df.write('\n')           
                df.close()
            
            self.origin_graphs_comp1 = origin_graphs_comp1
            self.origin_graphs_comp2 = origin_graphs_comp2
            _label_values = self.df['value']
            _label_Tb_comp1 = self.df['Tb_comp1']
            _label_Tb_comp2 = self.df['Tb_comp2']
            _label_Tc_comp1 = self.df['Tc_comp1']
            _label_Tc_comp2 = self.df['Tc_comp2']
            self.values = F.zerocopy_from_numpy(np.nan_to_num(_label_values).astype(np.float32))[self.valid_idx]
            self.Tb_comp1 = F.zerocopy_from_numpy(np.nan_to_num(_label_Tb_comp1).astype(np.float32))[self.valid_idx]
            self.Tb_comp2 = F.zerocopy_from_numpy(np.nan_to_num(_label_Tb_comp2).astype(np.float32))[self.valid_idx]
            self.Tc_comp1 = F.zerocopy_from_numpy(np.nan_to_num(_label_Tc_comp1).astype(np.float32))[self.valid_idx]
            self.Tc_comp2 = F.zerocopy_from_numpy(np.nan_to_num(_label_Tc_comp2).astype(np.float32))[self.valid_idx]
            valid_idx = torch.tensor(self.valid_idx)
            save_graphs(cache_comp1_path, self.origin_graphs_comp1, labels={'values': self.values, 'valid_idx': valid_idx, 'Tb_comp1':self.Tb_comp1, 'Tc_comp1':self.Tc_comp1})
            save_graphs(cache_comp2_path, self.origin_graphs_comp2, labels={'values': self.values, 'valid_idx': valid_idx, 'Tb_comp2':self.Tb_comp2, 'Tc_comp2':self.Tc_comp2})
    
        self.smiles1 = [self.df['comp1'][i] for i in self.valid_idx]
        self.smiles2 = [self.df['comp2'][i] for i in self.valid_idx]
    
    def _prepare_frag(self, params, fragmentation, load, log_every, error_log):
       
        _frag_cache_file_path_comp1 = os.path.join(self.cache_file_path, params['Dataset'] +  '_comp1_' + self.name + '_frag')
        _frag_cache_file_path_comp2 = os.path.join(self.cache_file_path, params['Dataset'] +  '_comp2_' + self.name + 'frag')
        _motif_cache_file_path_comp1 = os.path.join(self.cache_file_path, params['Dataset'] +  '_comp1_' + self.name + 'motif')
        _motif_cache_file_path_comp2 = os.path.join(self.cache_file_path, params['Dataset'] +  '_comp2_' + self.name + 'motif')

        if os.path.exists(_frag_cache_file_path_comp1) and os.path.exists(_motif_cache_file_path_comp1) and load:
            print('Loading saved fragments and graphs for comp1 ...')
            unbatched_frag_graphs_comp1, frag_label_dict_comp1 = load_graphs(_frag_cache_file_path_comp1)
            self.motif_graphs_comp1, motif_label_dict_comp1 = load_graphs(_motif_cache_file_path_comp1)
            frag_graph_idx_comp1 = frag_label_dict_comp1['frag_graph_idx'].detach().numpy().tolist()
            self.batched_frag_graphs_comp1 = self.batch_frag_graph(unbatched_frag_graphs_comp1, frag_graph_idx_comp1)
        else:
            print('Preparing fragmentation for comp1 ...')
            self.batched_frag_graphs_comp1 = []
            unbatched_frag_graphs_list_comp1 = []
            self.motif_graphs_comp1 = []
            self.atom_mask_list_comp1 = []
            self.frag_flag_list_comp1 = []
            for i, s in enumerate(self.df['comp1']):
                if (i + 1) % log_every == 0:
                    print('Currently proceeding fragmentation on molecule {:d}/{:d}'.format(i + 1, len(self)))
                try:
                    frag_graph, motif_graph, atom_mask, frag_flag = graph_2_frag(s, self.origin_graphs_comp1[i], fragmentation)
                except:
                    print('Failed to deal with  ', s)
                
                self.batched_frag_graphs_comp1.append(dgl.batch(frag_graph))
                unbatched_frag_graphs_list_comp1.append(frag_graph)
                self.motif_graphs_comp1.append(motif_graph)
                self.atom_mask_list_comp1.append(atom_mask)
                self.frag_flag_list_comp1.append(frag_flag)

            # Check failed fragmentation
            batched_frag_graphs_comp1 = []
            unbatched_frag_graphs_comp1 = []
            motif_graphs_comp1 = []
            frag_failed_smiles_comp1 = []
            for i, g in enumerate(self.motif_graphs_comp1):
                if g is not None:
                    motif_graphs_comp1.append(g)
                    batched_frag_graphs_comp1.append(self.batched_frag_graphs_comp1[i])
                    unbatched_frag_graphs_comp1.append(unbatched_frag_graphs_list_comp1[i])
                else:
                    frag_failed_smiles_comp1.append((i, self.df['comp1'][i]))
                    self.valid_idx.remove(i)

            if len(frag_failed_smiles_comp1) > 0:
                failed_idx, failed_smis = map(list, zip(*frag_failed_smiles_comp1))
            else:
                failed_idx, failed_smis = [], []
            df = pd.DataFrame({'raw_id': failed_idx, 'smiles': failed_smis})
            if os.path.exists(error_log):
                df.to_csv(error_log, mode='a', index=False)
            else:
                df.to_csv(error_log, mode='w', index=False)
            self.batched_frag_graphs_comp1 = batched_frag_graphs_comp1
            self.motif_graphs_comp1 = motif_graphs_comp1
            unbatched_frag_graphs_comp1, frag_graph_idx_comp1 = self.merge_frag_list(unbatched_frag_graphs_comp1)
            valid_idx = torch.tensor(self.valid_idx)
            save_graphs(_frag_cache_file_path_comp1, unbatched_frag_graphs_comp1, labels={'values': self.values, 'valid_idx': valid_idx, 'frag_graph_idx': frag_graph_idx_comp1})
            save_graphs(_motif_cache_file_path_comp1, self.motif_graphs_comp1, labels={'values': self.values, 'valid_idx': valid_idx})


        if os.path.exists(_frag_cache_file_path_comp2) and os.path.exists(_motif_cache_file_path_comp2) and load:
            unbatched_frag_graphs_comp2, frag_label_dict_comp2 = load_graphs(_frag_cache_file_path_comp2)
            self.motif_graphs_comp2, motif_label_dict_comp2 = load_graphs(_motif_cache_file_path_comp2)
            frag_graph_idx_comp2 = frag_label_dict_comp2['frag_graph_idx'].detach().numpy().tolist()
            self.batched_frag_graphs_comp2 = self.batch_frag_graph(unbatched_frag_graphs_comp2, frag_graph_idx_comp2)
        
        else:
            print('Preparing fragmentation for comp2 ...')
            self.batched_frag_graphs_comp2 = []
            unbatched_frag_graphs_list_comp2 = []
            self.motif_graphs_comp2 = []
            self.atom_mask_list_comp2 = []
            self.frag_flag_list_comp2 = []
            for i, s in enumerate(self.df['comp2']):
                if (i + 1) % log_every == 0:
                    print('Currently proceeding fragmentation on molecule {:d}/{:d}'.format(i + 1, len(self)))
                try:
                    frag_graph, motif_graph, atom_mask, frag_flag = graph_2_frag(s, self.origin_graphs_comp2[i], fragmentation)
                except:
                    print('Failed to deal with  ', s)
                
                self.batched_frag_graphs_comp2.append(dgl.batch(frag_graph))
                unbatched_frag_graphs_list_comp2.append(frag_graph)
                self.motif_graphs_comp2.append(motif_graph)
                self.atom_mask_list_comp2.append(atom_mask)
                self.frag_flag_list_comp2.append(frag_flag)

            # Check failed fragmentation
            batched_frag_graphs_comp2 = []
            unbatched_frag_graphs_comp2 = []
            motif_graphs_comp2 = []
            frag_failed_smiles_comp2 = []
            for i, g in enumerate(self.motif_graphs_comp2):
                if g is not None:
                    motif_graphs_comp2.append(g)
                    batched_frag_graphs_comp2.append(self.batched_frag_graphs_comp2[i])
                    unbatched_frag_graphs_comp2.append(unbatched_frag_graphs_list_comp2[i])
                else:
                    frag_failed_smiles_comp2.append((i, self.df['comp2'][i]))
                    self.valid_idx.remove(i)

            if len(frag_failed_smiles_comp2) > 0:
                failed_idx, failed_smis = map(list, zip(*frag_failed_smiles_comp2))
            else:
                failed_idx, failed_smis = [], []
            df = pd.DataFrame({'raw_id': failed_idx, 'smiles': failed_smis})
            if os.path.exists(error_log):
                df.to_csv(error_log, mode='a', index=False)
            else:
                df.to_csv(error_log, mode='w', index=False)
            self.batched_frag_graphs_comp2 = batched_frag_graphs_comp2
            self.motif_graphs_comp2 = motif_graphs_comp2
            unbatched_frag_graphs_comp2, frag_graph_idx_comp2 = self.merge_frag_list(unbatched_frag_graphs_comp2)
            valid_idx = torch.tensor(self.valid_idx)
            save_graphs(_frag_cache_file_path_comp2, unbatched_frag_graphs_comp2, labels={'values': self.values, 'valid_idx': valid_idx, 'frag_graph_idx': frag_graph_idx_comp1})
            save_graphs(_motif_cache_file_path_comp2, self.motif_graphs_comp2, labels={'values': self.values, 'valid_idx': valid_idx})
        
    def __len__(self):
        return len(self.df['comp1'])
    
    def __getitem__(self,index):
        if self.whe_frag:
            return self.origin_graphs_comp1[index], self.origin_graphs_comp2[index], self.batched_frag_graphs_comp1[index], self.batched_frag_graphs_comp2[index], self.motif_graphs_comp1[index], self.motif_graphs_comp2[index], self.channel_graphs_comp1[index], self.channel_graphs_comp2[index],self.values[index], self.smiles1[index],self.smiles2[index], self.Tb_comp1[index], self.Tc_comp1[index], self.Tb_comp2[index], self.Tc_comp2[index]
        else:
            return self.origin_graphs_comp1[index], self.origin_graphs_comp2[index], self.values[index], self.smiles1[index], self.smiles2[index], self.Tb_comp1[index], self.Tc_comp1[index], self.Tb_comp2[index], self.Tc_comp2[index]
        
    def _prepare_channel(self):
        self.channel_graphs_comp1 = []
        self.channel_graphs_comp2 = []
        for _ in range(len(self.df['comp1'])):
            self.channel_graphs_comp1.append(create_channels())#这个create_channels是啥意思
        for _ in range(len(self.df['comp1'])):
            self.channel_graphs_comp2.append(create_channels())
    
    
    def merge_frag_list(self, frag_graphs_list):
        # flatten all fragment lists in self.frag_graphs_lists for saving, [[...], [...], [...], ...] --> [..., ..., ...]
        frag_graphs = []
        idx = []
        for i, item in enumerate(frag_graphs_list):
            for _ in range(len(item)):
                idx.append(i)
            frag_graphs.extend(item)
        idx = torch.Tensor(idx)
        return frag_graphs, idx

    def batch_frag_graph(self, unbatched_graph, frag_graph_idx):
        batched_frag_graphs = []
        for i in range(len(self)):
            batched_frag_graph = dgl.batch([unbatched_graph[idx] for idx, value in enumerate(frag_graph_idx) if int(value) == i])
            batched_frag_graphs.append(batched_frag_graph)
        return batched_frag_graphs