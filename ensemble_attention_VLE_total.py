#this file is used for calculate the attention values of groups for certain VLE prediction model
from dataload.dataloading import import_dataset
from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.junctiontree_encoder import JT_SubGraph
from dataload.Azeotrope_dataset import Azeotrope_Dataset
from dataload.VLE_dataset import VLE_Dataset
from utils.mol2graph import smiles_2_bigraph
from utils.VLE_splitter import VLE_Splitter2
from src.dgltools import collate_fraggraphs_VLE2,collate_fraggraphs_pair
from utils.count_parameters import count_parameters
from utils.VLE_piplines import Azeotrope_PreFetch,predict_Tb_Tc,evaluate_frag_VLE_attention_value
from utils.piplines import  PreFetch

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

splitting_seed = [3630]
init_seed = [635]
position = 3
dataset_list = ['VLE_total_cleaned']
path_l = ['Tb_JCIM_normalized/all','Tc_JCIM_normalized/all']
name_l = ['Ensemble_0_Tb_JCIM_normalized_AGCNet','Ensemble_0_Tc_JCIM_normalized_AGCNet']
model_path = ['VLE_total_cleaned']
model_name = ['Ensemble_0_VLE_total_cleaned_AGCNetVLE']
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

    path = model_path[j]
    name = model_name[j]

    name_idx = name + '_' + str(position)
    params, net_params, model = load_model(path, name_idx)
    n_param = count_parameters(model)

    _, _, _, train_predict_T, train_target_T, train_predict_Y, train_target_Y, attention_comp1_train, attention_comp2_train, smiles_train_comp1, smiles_train_comp2\
            = evaluate_frag_VLE_attention_value(model,scaling_T, scaling_y1, fetched_data.train_iter, 
                                                                        fetched_data.train_batched_origin_graph_list_comp1, fetched_data.train_batched_origin_graph_list_comp2,
                                                                        fetched_data.train_batched_frag_graph_list_comp1, fetched_data.train_batched_frag_graph_list_comp2,
                                                                        fetched_data.train_batched_motif_graph_list_comp1, fetched_data.train_batched_motif_graph_list_comp2,
                                                                        fetched_data.train_Tb_comp1,
                                                                        fetched_data.train_Tc_comp1,
                                                                        fetched_data.train_Tb_comp2,
                                                                        fetched_data.train_Tc_comp2,
                                                                        fetched_data.train_x1,
                                                                        fetched_data.train_T,
                                                                        fetched_data.train_y1,
                                                                        fetched_data.train_smiles_list_comp1,
                                                                        fetched_data.train_smiles_list_comp2,n_param)
    
    
    _, _, _, val_predict_T, val_target_T, val_predict_Y, val_target_Y, \
    attention_comp1_val, attention_comp2_val, smiles_val_comp1, smiles_val_comp2= evaluate_frag_VLE_attention_value(model,scaling_T,scaling_y1,fetched_data.val_iter, 
                                                                        fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                        fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                        fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                        fetched_data.val_Tb_comp1,
                                                                        fetched_data.val_Tc_comp1,
                                                                        fetched_data.val_Tb_comp2,
                                                                        fetched_data.val_Tc_comp2,
                                                                        fetched_data.val_x1,
                                                                        fetched_data.val_T,
                                                                        fetched_data.val_y1,
                                                                        fetched_data.val_smiles_list_comp1,
                                                                        fetched_data.val_smiles_list_comp2,n_param)


    _, _, _, test_predict_T, test_target_T, test_predict_Y, test_target_Y, \
    attention_comp1_test, attention_comp2_test, smiles_test_comp1, smiles_test_comp2= evaluate_frag_VLE_attention_value(model,scaling_T,scaling_y1,fetched_data.test_iter, 
                                                                        fetched_data.test_batched_origin_graph_list_comp1, fetched_data.test_batched_origin_graph_list_comp2,
                                                                        fetched_data.test_batched_frag_graph_list_comp1, fetched_data.test_batched_frag_graph_list_comp2,
                                                                        fetched_data.test_batched_motif_graph_list_comp1, fetched_data.test_batched_motif_graph_list_comp2,
                                                                        fetched_data.test_Tb_comp1,
                                                                        fetched_data.test_Tc_comp1,
                                                                        fetched_data.test_Tb_comp2,
                                                                        fetched_data.test_Tc_comp2,
                                                                        fetched_data.test_x1,
                                                                        fetched_data.test_T,
                                                                        fetched_data.test_y1,
                                                                        fetched_data.test_smiles_list_comp1,
                                                                        fetched_data.test_smiles_list_comp2,n_param)
    
    save_dir = os.path.join('./library', path)
    df_train = pd.DataFrame({'SMILES_comp1': smiles_train_comp1[0], 'SMILES_comp2': smiles_train_comp2[0],
                            'Target_T': train_target_T.numpy().flatten().tolist(),
                            'Predict_T': train_predict_T.numpy().flatten().tolist(),
                            'Target_Y': train_target_Y.numpy().flatten().tolist(),
                            'Predict_Y': train_predict_Y.numpy().flatten().tolist(),
                            'attention_comp1':np.array([v.tolist() for v in attention_comp1_train],dtype = list),
                            'attention_comp2':np.array([v.tolist() for v in attention_comp2_train],dtype = list)})
    save_train_path = os.path.join(save_dir, '{}_{}'.format(dataset_list[j], time.strftime('%Y-%m-%d-%H-%M')) + 'train.csv')
    df_train.to_csv(save_train_path, index=False)

    df_val = pd.DataFrame({'SMILES_comp1': smiles_val_comp1[0], 'SMILES_comp2': smiles_val_comp2[0],
                            'Target_T': val_target_T.numpy().flatten().tolist(),
                            'Predict_T': val_predict_T.numpy().flatten().tolist(),
                            'Target_Y': val_target_Y.numpy().flatten().tolist(),
                            'Predict_Y': val_predict_Y.numpy().flatten().tolist(),                                
                            'attention_comp1':np.array([v.tolist() for v in attention_comp1_val],dtype = list),
                            'attention_comp2':np.array([v.tolist() for v in attention_comp2_val],dtype = list)})
    save_val_path = os.path.join(save_dir, '{}_{}'.format(dataset_list[j], time.strftime('%Y-%m-%d-%H-%M')) + 'val.csv')
    df_val.to_csv(save_val_path, index=False)

    df_test = pd.DataFrame({'SMILES_comp1': smiles_test_comp1[0], 'SMILES_comp2': smiles_test_comp2[0],
                            'Target_T': test_target_T.numpy().flatten().tolist(),
                            'Predict_T': test_predict_T.numpy().flatten().tolist(),
                            'Target_Y': test_target_Y.numpy().flatten().tolist(),
                            'Predict_Y': test_predict_Y.numpy().flatten().tolist(),                                
                            'attention_comp1':np.array([v.tolist() for v in attention_comp1_test],dtype = list),
                            'attention_comp2':np.array([v.tolist() for v in attention_comp2_test],dtype = list)})
    
    save_test_path = os.path.join(save_dir, '{}_{}'.format(dataset_list[j], time.strftime('%Y-%m-%d-%H-%M')) + 'test.csv')
    df_test.to_csv(save_test_path, index=False)

    
