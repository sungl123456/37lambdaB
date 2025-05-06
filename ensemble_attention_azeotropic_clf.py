from dataload.dataloading import import_dataset
from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.junctiontree_encoder import JT_SubGraph
from dataload.Azeotrope_dataset import Azeotrope_Dataset
from dataload.pair_dataset import pair_Dataset
from utils.mol2graph import smiles_2_bigraph
from utils.VLE_splitter import Azeotrope_Splitter, pair_Splitter3
from src.dgltools import collate_fraggraphs_VLE,collate_fraggraphs_pair
from utils.Earlystopping import EarlyStopping
from utils.count_parameters import count_parameters
from utils.VLE_piplines import Azeotrope_PreFetch,predict_Tb_Tc,evaluate_frag_clf_attention_value2
from utils.piplines import train_epoch, evaluate, train_epoch_frag, evaluate_frag, PreFetch, evaluate_frag_descriptors, evaluate_frag_attention
from utils.Set_Seed_Reproducibility import set_seed

import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from networks.DMPNN import DMPNNNet
from networks.MPNN import MPNNNet
from networks.AttentiveFP import AttentiveFPNet
from networks.FraGAT import NewFraGATNet
from networks.AGC import AGCNet
from networks.AGC_VLE import AGCNetVLE
from dataload.model_library import save_model, load_model, load_optimal_model

params = {}
net_params = {}
params['init_lr'] = 10 ** -2
params['min_lr'] = 10 ** -9
params['weight_decay'] = 10 ** -2
params['lr_reduce_factor'] = 0.75
params['lr_schedule_patience'] = 30
params['earlystopping_patience'] = 100
params['max_epoch'] = 300

net_params['num_atom_type'] = 36
net_params['num_bond_type'] = 12
net_params['hidden_dim'] = 16
net_params['num_heads'] = 1
net_params['dropout'] = 0
net_params['depth'] = 2
net_params['layers'] = 2
net_params['residual'] = False
net_params['batch_norm'] = False
net_params['layer_norm'] = False
net_params['device'] = 'cpu'
splitting_seed = 336
init_seed = 101
position = 4
dataset_list = ['azeotrope_classification']
path_l = ['Tb_JCIM_normalized/all','Tc_JCIM_normalized/all']
name_l = ['Ensemble_0_Tb_JCIM_normalized_AGCNet','Ensemble_0_Tc_JCIM_normalized_AGCNet']
model_path = ['azeotrope_classification']
model_name = ['Ensemble_0_azeotrope_classification_AGCNetVLE']

params['Dataset'] = dataset_list[0]
net_params['Dataset'] = dataset_list[0]
df = import_dataset(params)

cache_file_path = os.path.realpath('./cache')
if not os.path.exists(cache_file_path):
        os.mkdir(cache_file_path)
cache_file =  os.path.join(cache_file_path, params['Dataset'] + '_')

error_path = os.path.realpath('./error_log')
if not os.path.exists(error_path):
    os.mkdir(error_path)
error_log_path = os.path.join(error_path, '{}_{}'.format(params['Dataset'], time.strftime('%Y-%m-%d-%H-%M')) + '.csv')

# fragmentation = frag(frag_doc='MG_plus_reference', ring_doc = 'Ring_structure2')
fragmentation = JT_SubGraph(scheme = 'MG_plus_reference')
net_params['frag_dim'] = fragmentation.frag_dim

allset = Azeotrope_Dataset(df = df, params = params, name = 'all',
                                smiles_2_graph = smiles_2_bigraph,
                                atom_featurizer = classic_atom_featurizer,
                                bond_featurizer = classic_bond_featurizer,
                                mol_featurizer = classic_mol_featurizer, 
                                cache_file_path = cache_file,
                                error_log=error_log_path,
                                load = False,
                                fragmentation=fragmentation)

raw_loader = DataLoader(allset,collate_fn=collate_fraggraphs_pair, batch_size=len(allset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))

fetched_data = Azeotrope_PreFetch(train_loader=None, val_loader=None, test_loader=None, raw_loader=raw_loader, frag=2)

path_Tb = path_l[0]
name_Tb = name_l[0]
_, _, model_Tb = load_model(path_Tb, name_Tb +'_0')
path_Tc = path_l[1]
name_Tc = name_l[1]
_, _, model_Tc = load_model(path_Tc, name_Tc +'_0')

comp1_Tb, comp2_Tb, comp1_Tc, comp2_Tc = predict_Tb_Tc(model_Tb,model_Tc,fetched_data.all_iter, 
                                            fetched_data.all_batched_origin_graph_list_comp1, fetched_data.all_batched_origin_graph_list_comp2,
                                            fetched_data.all_batched_frag_graph_list_comp1, fetched_data.all_batched_frag_graph_list_comp2,
                                            fetched_data.all_batched_motif_graph_list_comp1, fetched_data.all_batched_motif_graph_list_comp2)
comp1_Tb = pd.DataFrame(comp1_Tb)
comp2_Tb = pd.DataFrame(comp2_Tb)
comp1_Tc = pd.DataFrame(comp1_Tc)
comp2_Tc = pd.DataFrame(comp2_Tc)
df['comp1_Tb'] = comp1_Tb
df['comp2_Tb'] = comp2_Tb
df['comp1_Tc'] = comp1_Tc
df['comp2_Tc'] = comp2_Tc

torch.manual_seed(init_seed)

comp = np.array(df)
all_size = len(comp)

comp = np.array(df)
all_size = len(comp)
comp_all = pd.DataFrame(comp,columns=['comp1','comp2','value','Tb_comp1','Tb_comp2','Tc_comp1','Tc_comp2'])
dataset = pair_Dataset(df = comp_all, params = params, name = 'all',
                                smiles_2_graph = smiles_2_bigraph,
                                atom_featurizer = classic_atom_featurizer,
                                bond_featurizer = classic_bond_featurizer,
                                mol_featurizer = classic_mol_featurizer, 
                                cache_file_path = cache_file,
                                error_log=error_log_path,
                                load = False,
                                fragmentation=fragmentation)

split = pair_Splitter3(dataset)
set_seed(seed = 1000)
torch.manual_seed(init_seed)
trainset, valset, testset, allset = split.Random_Splitter(comp, seed = splitting_seed, frac_train = 0.8, frac_val = 0.9)
train_loader = DataLoader(trainset, collate_fn=collate_fraggraphs_VLE, batch_size=len(trainset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
val_loader = DataLoader(valset,collate_fn=collate_fraggraphs_VLE, batch_size=len(valset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
test_loader = DataLoader(testset,collate_fn=collate_fraggraphs_VLE, batch_size=len(testset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
raw_loader = DataLoader(allset,collate_fn=collate_fraggraphs_VLE, batch_size=len(allset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))

fetched_data = Azeotrope_PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=3)

path = model_path[0]
name = model_name[0]

name_idx = name + '_' + str(position)
params, net_params, model = load_model(path, name_idx)
n_param = count_parameters(model)


_, _, test_predict, test_target,\
test_attention_comp1_list, test_attention_comp2_list, \
test_smiles_list_comp1, test_smiles_list_comp2,test_score_list  = evaluate_frag_clf_attention_value2(model,fetched_data.test_iter, 
                                                                    fetched_data.test_batched_origin_graph_list_comp1,
                                                                    fetched_data.test_batched_origin_graph_list_comp2,
                                                                    fetched_data.test_batched_frag_graph_list_comp1, 
                                                                    fetched_data.test_batched_frag_graph_list_comp2,
                                                                    fetched_data.test_batched_motif_graph_list_comp1, 
                                                                    fetched_data.test_batched_motif_graph_list_comp2,
                                                                    fetched_data.test_Tb_comp1,
                                                                    fetched_data.test_Tc_comp1,
                                                                    fetched_data.test_Tb_comp2,
                                                                    fetched_data.test_Tc_comp2,
                                                                    fetched_data.test_targets_list,
                                                                    fetched_data.test_smiles_list_comp1,
                                                                    fetched_data.test_smiles_list_comp2)

_, _, all_predict, all_target,\
attention_comp1_list, attention_comp2_list, \
smiles_list_comp1, smiles_list_comp2,all_score_list  = evaluate_frag_clf_attention_value2(model,fetched_data.all_iter, 
                                                                    fetched_data.all_batched_origin_graph_list_comp1, fetched_data.all_batched_origin_graph_list_comp2,
                                                                    fetched_data.all_batched_frag_graph_list_comp1, fetched_data.all_batched_frag_graph_list_comp2,
                                                                    fetched_data.all_batched_motif_graph_list_comp1, fetched_data.all_batched_motif_graph_list_comp2,
                                                                    fetched_data.all_Tb_comp1,
                                                                    fetched_data.all_Tc_comp1,
                                                                    fetched_data.all_Tb_comp2,
                                                                    fetched_data.all_Tc_comp2,
                                                                    fetched_data.all_targets_list,
                                                                    fetched_data.all_smiles_list_comp1,
                                                                    fetched_data.all_smiles_list_comp2)

df_test_results = pd.DataFrame({'SMILES_comp1': test_smiles_list_comp1[0], 'SMILES_comp2': test_smiles_list_comp2[0],
                        'Target': test_target.numpy().flatten().tolist(),
                        'Predict': test_predict.numpy().flatten().tolist(),
                        'attention_comp1':np.array([v.tolist() for v in test_attention_comp1_list],dtype = list),
                        'attention_comp2':np.array([v.tolist() for v in test_attention_comp2_list],dtype = list)})

df_results = pd.DataFrame({'SMILES_comp1': smiles_list_comp1[0], 'SMILES_comp2': smiles_list_comp2[0],
                        'Target': all_target.numpy().flatten().tolist(),
                        'Predict': all_predict.numpy().flatten().tolist(),
                        'attention_comp1':np.array([v.tolist() for v in attention_comp1_list],dtype = list),
                        'attention_comp2':np.array([v.tolist() for v in attention_comp2_list],dtype = list)})
# df_results = pd.concat([df_value, df_attention])
# op_idx, init_seed, seed, params, net_params, model = load_optimal_model(path, name)

# save_train_file_path = os.path.join('./library/' + path, '{}_{}_{}_{}_{}'.format('train',name, dataset_list[0], splitting_seed, time.strftime('%Y-%m-%d-%H-%M')) + '.csv')
save_test_file_path = os.path.join('./library/' + path, '{}_{}_{}_{}_{}'.format('test',name, dataset_list[0], splitting_seed, time.strftime('%Y-%m-%d-%H-%M')) + '.csv')
save_file_path = os.path.join('./library/' + path, '{}_{}_{}_{}_{}'.format('all',name, dataset_list[0], splitting_seed, time.strftime('%Y-%m-%d-%H-%M')) + '.csv')
# df_train_results.to_csv(save_train_file_path, index = False)
df_test_results.to_csv(save_test_file_path,index= False)
df_results.to_csv(save_file_path, index=False)




