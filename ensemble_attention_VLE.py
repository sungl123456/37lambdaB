from dataload.dataloading import import_dataset
from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.junctiontree_encoder import JT_SubGraph
from dataload.Azeotrope_dataset import Azeotrope_Dataset
from dataload.pair_dataset import pair_Dataset
from dataload.VLE_dataset import VLE_Dataset
from utils.mol2graph import smiles_2_bigraph
from utils.VLE_splitter import Azeotrope_Splitter, pair_Splitter2, VLE_Splitter2
from src.dgltools import collate_fraggraphs_VLE2,collate_fraggraphs_pair
from utils.Earlystopping import EarlyStopping
from utils.count_parameters import count_parameters
from utils.VLE_piplines import Azeotrope_PreFetch,predict_Tb_Tc,evaluate_frag_VLE_attention_value4
from utils.piplines import train_epoch, evaluate, train_epoch_frag, evaluate_frag, PreFetch, evaluate_frag_descriptors, evaluate_frag_attention

import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from dataload.model_library import load_model

params = {}
net_params = {}
params['init_lr'] = 10 ** -3
params['min_lr'] = 1e-9
params['weight_decay'] = 10 ** -2
params['lr_reduce_factor'] = 0.7
params['lr_schedule_patience'] = 30
params['earlystopping_patience'] = 100
params['max_epoch'] = 300
params['lam_T'] = 0.59
params['lam_Y'] = 0.46

net_params['num_atom_type'] = 36
net_params['num_bond_type'] = 12
net_params['hidden_dim'] = 169
net_params['num_heads'] = 1
net_params['dropout'] = 0
net_params['depth'] = 5
net_params['layers'] = 1
net_params['residual'] = False
net_params['batch_norm'] = False
net_params['layer_norm'] = False
net_params['device'] = 'cpu'
splitting_seed = [2931,3777]
init_seed = [361,361]
model_num = [0,0]
dataset_list = ['VLE_zeotrope','VLE_azeotrope']
path = ['VLE_zeotrope','VLE_azeotrope']
path_l = ['Tb_JCIM_normalized/all','Tc_JCIM_normalized/all']
name_l = ['Ensemble_0_Tb_JCIM_normalized_AGCNet','Ensemble_0_Tc_JCIM_normalized_AGCNet']
model_path_T = ['VLE_zeotrope/T','VLE_azeotrope/T']
model_name_T = ['Ensemble_0_VLE_zeotrope_AGCNetVLE2','Ensemble_0_VLE_azeotrope_AGCNetVLE2']
model_path_Y = ['VLE_zeotrope/Y','VLE_azeotrope/Y']
model_name_Y = ['Ensemble_0_VLE_zeotrope_AGCNetVLE3','Ensemble_0_VLE_azeotrope_AGCNetVLE3']

for j in range(len(dataset_list)):
    params['Dataset'] = dataset_list[j]
    net_params['Dataset'] = dataset_list[j]
    df, scaling_T, scaling_y1= import_dataset(params)
    df['value']  = df['T']
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
                                    load = True,
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
    del df['value']
    torch.manual_seed(init_seed[j])

    comp = np.array(df)
    all_size = len(comp)

    # Contents of train_set, val_set will be disrupted
    comp_all = pd.DataFrame(comp,columns=['comp1','comp2','x1','T','y1','Tb_comp1','Tb_comp2','Tc_comp1','Tc_comp2'])

    allset = VLE_Dataset(df = comp_all, params = params, name = 'all',
                                    smiles_2_graph = smiles_2_bigraph,
                                    atom_featurizer = classic_atom_featurizer,
                                    bond_featurizer = classic_bond_featurizer,
                                    mol_featurizer = classic_mol_featurizer, 
                                    cache_file_path = cache_file,
                                    error_log=error_log_path,
                                    load = True,
                                    fragmentation=fragmentation)
    
    split = VLE_Splitter2(allset)
    torch.manual_seed(init_seed[j])
    train_dataset, val_dataset, test_dataset, all_dataset = split.Random_Splitter(comp, seed = 100, frac_train = 0.8, frac_val = 0.9)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fraggraphs_VLE2, batch_size=len(train_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    val_loader = DataLoader(val_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(val_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    test_loader = DataLoader(test_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(test_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    raw_loader = DataLoader(all_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(all_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))

    fetched_data = Azeotrope_PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=4)

    path_T = model_path_T[j]
    name_T = model_name_T[j]

    path_Y = model_path_Y[j]
    name_Y = model_name_Y[j]

    name_idx_T = name_T + '_' + str(model_num[j])
    params_T, net_params_T, model_T = load_model(path_T, name_idx_T)
    n_param_T = count_parameters(model_T)

    name_idx_Y = name_Y + '_' + str(model_num[j])
    params_Y, net_params_Y, model_Y = load_model(path_Y, name_idx_Y)
    n_param_Y = count_parameters(model_Y)

    _, _, test_predict_T, test_target_T,\
    attention_comp1_list_T, attention_comp2_list_T, \
    smiles_list_comp1_T, smiles_list_comp2_T  = evaluate_frag_VLE_attention_value4(model_T,scaling_T,fetched_data.test_iter, 
                                                                        fetched_data.test_batched_origin_graph_list_comp1, fetched_data.test_batched_origin_graph_list_comp2,
                                                                        fetched_data.test_batched_frag_graph_list_comp1, fetched_data.test_batched_frag_graph_list_comp2,
                                                                        fetched_data.test_batched_motif_graph_list_comp1, fetched_data.test_batched_motif_graph_list_comp2,
                                                                        fetched_data.test_Tb_comp1,
                                                                        fetched_data.test_Tc_comp1,
                                                                        fetched_data.test_Tb_comp2,
                                                                        fetched_data.test_Tc_comp2,
                                                                        fetched_data.test_x1,
                                                                        fetched_data.test_T,
                                                                        fetched_data.test_smiles_list_comp1,
                                                                        fetched_data.test_smiles_list_comp2,n_param_T,params['lam_T'])
    
    _, _, test_predict_Y, test_target_Y,\
    attention_comp1_list_Y, attention_comp2_list_Y, \
    smiles_list_comp1_Y, smiles_list_comp2_Y  = evaluate_frag_VLE_attention_value4(model_Y,scaling_y1,fetched_data.test_iter, 
                                                                        fetched_data.test_batched_origin_graph_list_comp1, fetched_data.test_batched_origin_graph_list_comp2,
                                                                        fetched_data.test_batched_frag_graph_list_comp1, fetched_data.test_batched_frag_graph_list_comp2,
                                                                        fetched_data.test_batched_motif_graph_list_comp1, fetched_data.test_batched_motif_graph_list_comp2,
                                                                        fetched_data.test_Tb_comp1,
                                                                        fetched_data.test_Tc_comp1,
                                                                        fetched_data.test_Tb_comp2,
                                                                        fetched_data.test_Tc_comp2,
                                                                        fetched_data.test_x1,
                                                                        fetched_data.test_y1,
                                                                        fetched_data.test_smiles_list_comp1,
                                                                        fetched_data.test_smiles_list_comp2,n_param_Y,params['lam_Y'])

    df_value_T = pd.DataFrame({'SMILES_comp1': smiles_list_comp1_T[0], 'SMILES_comp2': smiles_list_comp2_T[0],
                            'Target': test_target_T.numpy().flatten().tolist(),
                            'Predict': test_predict_T.numpy().flatten().tolist(),
                            'attention_comp1':np.array([v.tolist() for v in attention_comp1_list_T],dtype = list),
                            'attention_comp2':np.array([v.tolist() for v in attention_comp2_list_T],dtype = list)})
    
    save_file_path_T = os.path.join('./library/' + path[j], '{}_{}'.format(dataset_list[j], time.strftime('%Y-%m-%d-%H-%M')) + '_T.csv')
    df_value_T.to_csv(save_file_path_T, index=False)

    df_value_Y= pd.DataFrame({'SMILES_comp1': smiles_list_comp1_Y[0], 'SMILES_comp2': smiles_list_comp2_Y[0],
                            'Target': test_target_Y.numpy().flatten().tolist(),
                            'Predict': test_predict_Y.numpy().flatten().tolist(),                                
                            'attention_comp1':np.array([v.tolist() for v in attention_comp1_list_Y],dtype = list),
                            'attention_comp2':np.array([v.tolist() for v in attention_comp2_list_Y],dtype = list)})
    save_file_path_Y = os.path.join('./library/' + path[j], '{}_{}'.format(dataset_list[j], time.strftime('%Y-%m-%d-%H-%M')) + '_Y.csv')
    df_value_Y.to_csv(save_file_path_Y, index=False)

    
