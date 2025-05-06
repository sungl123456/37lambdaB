
from dataload.dataloading import import_dataset
from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.junctiontree_encoder import JT_SubGraph
from dataload.Azeotrope_dataset import Azeotrope_Dataset
from dataload.VLE_dataset import VLE_Dataset
from utils.mol2graph import smiles_2_bigraph
from utils.VLE_splitter import VLE_Splitter2
from src.dgltools import collate_fraggraphs_VLE2, collate_fraggraphs_pair
from utils.Earlystopping import EarlyStopping
from utils.count_parameters import count_parameters
from utils.VLE_piplines import Azeotrope_PreFetch,train_epoch_frag_VLE,predict_Tb_Tc, evaluate_frag_VLE
from utils.Set_Seed_Reproducibility import set_seed

import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from networks.AGC_VLE2 import AGCNetVLE2
from networks.AGC_VLE3 import AGCNetVLE3
from dataload.model_library import load_model

params = {}
net_params = {}

params['init_lr'] = 10 ** -2
params['min_lr'] = 1e-9
params['weight_decay'] = 0
params['lr_reduce_factor'] = 0.8
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
dataset_list = ['VLE_zeotrope','VLE_azeotrope']
path_l = ['Tb_JCIM_normalized/all','Tc_JCIM_normalized/all']
name_l = ['Ensemble_0_Tb_JCIM_normalized_AGCNet','Ensemble_0_Tc_JCIM_normalized_AGCNet']

     
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

    checkpoint_path = os.path.realpath('./checkpoint')
    if not os.path.exists(checkpoint_path):
         os.mkdir(checkpoint_path)
         
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
    df['comp1_Tb'] = pd.DataFrame(comp1_Tb)
    df['comp2_Tb'] = pd.DataFrame(comp2_Tb)
    df['comp1_Tc'] = pd.DataFrame(comp1_Tc)
    df['comp2_Tc'] = pd.DataFrame(comp2_Tc)
    del df['value']

    comp = np.array(df)
    all_size = len(comp)
    comp_all = pd.DataFrame(comp,columns=['comp1','comp2','x1','T','y1','Tb_comp1','Tb_comp2','Tc_comp1','Tc_comp2'])
    dataset = VLE_Dataset(df = comp_all, params = params, name = 'all',
                                    smiles_2_graph = smiles_2_bigraph,
                                    atom_featurizer = classic_atom_featurizer,
                                    bond_featurizer = classic_bond_featurizer,
                                    mol_featurizer = classic_mol_featurizer, 
                                    cache_file_path = cache_file,
                                    error_log=error_log_path,
                                    load = False,
                                    fragmentation=fragmentation)
    
    split = VLE_Splitter2(dataset)

    rows = []
    file_path = os.path.realpath('./output')
    if not os.path.exists(file_path):
         os.mkdir(file_path)
    save_file_path = os.path.join(file_path,'{}_{}_{}'.format(params['Dataset'], 'AGCNet_Pair', time.strftime('%Y-%m-%d-%H-%M'))+'.csv')
    wr = pd.DataFrame(columns=['seed', 'train_T_R2', 'val_T_R2', 'test_T_R2', 'all_T_R2', 
                            'train_y1_R2', 'val_y1_R2', 'test_y1_R2', 'all_y1_R2',
                            'train_T_MAE', 'val_T_MAE', 'test_T_MAE', 'all_T_MAE',
                            'train_T_RMSE', 'val_T_RMSE', 'test_T_RMSE', 'all_T_RMSE',
                            'train_y1_MAE', 'val_y1_MAE', 'test_y1_MAE', 'all_y1_MAE',
                            'train_y1_RMSE', 'val_y1_RMSE', 'test_y1_RMSE', 'all_y1_RMSE'])
    for i in range(20):
        seed = np.random.randint(1,5000)
        set_seed(seed = 1000)
        train_dataset, val_dataset, test_dataset, all_dataset = split.Random_Splitter(comp, seed = seed, frac_train = 0.8, frac_val = 0.9)

        train_loader = DataLoader(train_dataset, collate_fn=collate_fraggraphs_VLE2, batch_size=len(train_dataset), shuffle=False)
        val_loader = DataLoader(val_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(val_dataset), shuffle=False)
        test_loader = DataLoader(test_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(test_dataset), shuffle=False)
        raw_loader = DataLoader(all_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(all_dataset), shuffle=False)

        fetched_data = Azeotrope_PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=4)

        modelT = AGCNetVLE2(net_params).to(device='cpu')
        optimizerT = torch.optim.Adam(modelT.parameters(), lr = params['init_lr'])
        schedulerT = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerT, mode='min', factor=params['lr_reduce_factor'],
                                                                    patience=params['lr_schedule_patience'])
        modelY = AGCNetVLE3(net_params).to(device='cpu')
        optimizerY = torch.optim.Adam(modelY.parameters(), lr = params['init_lr'])
        schedulerY = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerY, mode='min', factor=params['lr_reduce_factor'],
                                                                    patience=params['lr_schedule_patience'])
        t0 = time.time()
        per_epoch_time = []
        early_stoppingT = EarlyStopping(patience=params['earlystopping_patience'], path='./checkpoint/checkpoint_seed_' + params['Dataset'] + '_AGCNet_VLE2' + '_seek_T_' + '.pt')
        early_stoppingY = EarlyStopping(patience=params['earlystopping_patience'], path='./checkpoint/checkpoint_seed_' + params['Dataset'] + '_AGCNet_VLE2' + '_seek_Y_' + '.pt')
        
        with tqdm(range(params['max_epoch'])) as t:
            n_paramT = count_parameters(modelT)
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                modelT, epoch_train_loss, epoch_train_metrics  = train_epoch_frag_VLE(modelT, optimizerT, scaling_T,
                                                                                    fetched_data.train_iter,
                                                                                    fetched_data.train_batched_origin_graph_list_comp1,
                                                                                    fetched_data.train_batched_origin_graph_list_comp2,
                                                                                    fetched_data.train_batched_frag_graph_list_comp1,
                                                                                    fetched_data.train_batched_frag_graph_list_comp2,
                                                                                    fetched_data.train_batched_motif_graph_list_comp1,
                                                                                    fetched_data.train_batched_motif_graph_list_comp2,
                                                                                    fetched_data.train_Tb_comp1,
                                                                                    fetched_data.train_Tc_comp1,
                                                                                    fetched_data.train_Tb_comp2,
                                                                                    fetched_data.train_Tc_comp2,
                                                                                    fetched_data.train_x1,
                                                                                    fetched_data.train_T,
                                                                                    n_paramT)
                epoch_val_loss, epoch_val_metrics = evaluate_frag_VLE(modelT, scaling_T, fetched_data.val_iter, 
                                                                                fetched_data.val_batched_origin_graph_list_comp1,
                                                                                fetched_data.val_batched_origin_graph_list_comp2,
                                                                                fetched_data.val_batched_frag_graph_list_comp1, 
                                                                                fetched_data.val_batched_frag_graph_list_comp2,
                                                                                fetched_data.val_batched_motif_graph_list_comp1,
                                                                                fetched_data.val_batched_motif_graph_list_comp2,
                                                                                fetched_data.val_Tb_comp1,
                                                                                fetched_data.val_Tc_comp1,
                                                                                fetched_data.val_Tb_comp2,
                                                                                fetched_data.val_Tc_comp2,
                                                                                fetched_data.val_x1,
                                                                                fetched_data.val_T,
                                                                                fetched_data.val_smiles_list_comp1,fetched_data.val_smiles_list_comp2,
                                                                                n_paramT)
                epoch_test_loss, epoch_test_metrics = evaluate_frag_VLE(modelT, scaling_T, fetched_data.test_iter, 
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
                                                                                fetched_data.test_x1,
                                                                                fetched_data.test_T,fetched_data.test_smiles_list_comp1,fetched_data.test_smiles_list_comp2,
                                                                                n_paramT)
                
                t.set_postfix({'time': time.time() - start, 'lr': optimizerT.param_groups[0]['lr'],
                    'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss, 
                    'train_R2': epoch_train_metrics.R2, 'val_R2': epoch_val_metrics.R2})
                per_epoch_time.append(time.time() - start)

                schedulerT.step(epoch_val_loss)
                if optimizerT.param_groups[0]['lr'] < params['min_lr']:
                    print('\n! LR equal to min LR set.')
                    break

                early_stoppingT(epoch_val_loss, modelT)
                if early_stoppingT.early_stop:
                    break

        with tqdm(range(params['max_epoch'])) as t:
            n_paramY = count_parameters(modelY)
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                modelY, epoch_train_loss, epoch_train_metrics  = train_epoch_frag_VLE(modelY, optimizerY, scaling_y1,
                                                                                            fetched_data.train_iter,
                                                                                            fetched_data.train_batched_origin_graph_list_comp1,
                                                                                            fetched_data.train_batched_origin_graph_list_comp2,
                                                                                            fetched_data.train_batched_frag_graph_list_comp1,
                                                                                            fetched_data.train_batched_frag_graph_list_comp2,
                                                                                            fetched_data.train_batched_motif_graph_list_comp1,
                                                                                            fetched_data.train_batched_motif_graph_list_comp2,
                                                                                            fetched_data.train_Tb_comp1,
                                                                                            fetched_data.train_Tc_comp1,
                                                                                            fetched_data.train_Tb_comp2,
                                                                                            fetched_data.train_Tc_comp2,
                                                                                            fetched_data.train_x1,
                                                                                            fetched_data.train_y1,
                                                                                            n_paramY)
                

                epoch_val_loss, epoch_val_metrics = evaluate_frag_VLE(modelY, scaling_y1, fetched_data.val_iter, 
                                                                                fetched_data.val_batched_origin_graph_list_comp1,
                                                                                fetched_data.val_batched_origin_graph_list_comp2,
                                                                                fetched_data.val_batched_frag_graph_list_comp1, 
                                                                                fetched_data.val_batched_frag_graph_list_comp2,
                                                                                fetched_data.val_batched_motif_graph_list_comp1,
                                                                                fetched_data.val_batched_motif_graph_list_comp2,
                                                                                fetched_data.val_Tb_comp1,
                                                                                fetched_data.val_Tc_comp1,
                                                                                fetched_data.val_Tb_comp2,
                                                                                fetched_data.val_Tc_comp2,
                                                                                fetched_data.val_x1,
                                                                                fetched_data.val_y1,fetched_data.val_smiles_list_comp1,fetched_data.val_smiles_list_comp2,
                                                                                n_paramY)
                
                epoch_test_loss, epoch_test_metrics = evaluate_frag_VLE(modelY, scaling_y1, fetched_data.test_iter, 
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
                                                                                fetched_data.test_x1,
                                                                                fetched_data.test_y1,fetched_data.test_smiles_list_comp1,fetched_data.test_smiles_list_comp2,
                                                                                n_paramY)
                
                t.set_postfix({'time': time.time() - start, 'lr': optimizerY.param_groups[0]['lr'],
                    'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss, 
                    'train_R2': epoch_train_metrics.R2, 'val_R2': epoch_val_metrics.R2})
                per_epoch_time.append(time.time() - start)

                schedulerY.step(epoch_val_loss)
                if optimizerT.param_groups[0]['lr'] < params['min_lr']:
                    print('\n! LR equal to min LR set.')
                    break

                early_stoppingY(epoch_val_loss, modelY)
                if early_stoppingY.early_stop:
                    break

        modelT = early_stoppingT.load_checkpoint(modelT)

        _, epoch_train_T_metrics = evaluate_frag_VLE(modelT, scaling_T,fetched_data.train_iter,
                                                                            fetched_data.train_batched_origin_graph_list_comp1,
                                                                            fetched_data.train_batched_origin_graph_list_comp2,
                                                                            fetched_data.train_batched_frag_graph_list_comp1,
                                                                            fetched_data.train_batched_frag_graph_list_comp2,
                                                                            fetched_data.train_batched_motif_graph_list_comp1,
                                                                            fetched_data.train_batched_motif_graph_list_comp2,
                                                                            fetched_data.train_Tb_comp1,
                                                                            fetched_data.train_Tc_comp1,
                                                                            fetched_data.train_Tb_comp2,
                                                                            fetched_data.train_Tc_comp2,
                                                                            fetched_data.train_x1,
                                                                            fetched_data.train_T,fetched_data.train_smiles_list_comp1,fetched_data.train_smiles_list_comp2,n_paramT)
        _, epoch_val_T_metrics = evaluate_frag_VLE(modelT,scaling_T,fetched_data.val_iter, 
                                                                            fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                            fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                            fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                            fetched_data.val_Tb_comp1,
                                                                            fetched_data.val_Tc_comp1,
                                                                            fetched_data.val_Tb_comp2,
                                                                            fetched_data.val_Tc_comp2,
                                                                            fetched_data.val_x1,
                                                                            fetched_data.val_T,fetched_data.val_smiles_list_comp1,fetched_data.val_smiles_list_comp2,n_paramT)
        _, epoch_test_T_metrics = evaluate_frag_VLE(modelT,scaling_T,fetched_data.test_iter, 
                                                                            fetched_data.test_batched_origin_graph_list_comp1, fetched_data.test_batched_origin_graph_list_comp2,
                                                                            fetched_data.test_batched_frag_graph_list_comp1, fetched_data.test_batched_frag_graph_list_comp2,
                                                                            fetched_data.test_batched_motif_graph_list_comp1, fetched_data.test_batched_motif_graph_list_comp2,
                                                                            fetched_data.test_Tb_comp1,
                                                                            fetched_data.test_Tc_comp1,
                                                                            fetched_data.test_Tb_comp2,
                                                                            fetched_data.test_Tc_comp2,
                                                                            fetched_data.test_x1,
                                                                            fetched_data.test_T,fetched_data.test_smiles_list_comp1,fetched_data.test_smiles_list_comp2,n_paramT)
        _, epoch_raw_T_metrics = evaluate_frag_VLE(modelT,scaling_T,fetched_data.all_iter, 
                                                                            fetched_data.all_batched_origin_graph_list_comp1, fetched_data.all_batched_origin_graph_list_comp2,
                                                                            fetched_data.all_batched_frag_graph_list_comp1, fetched_data.all_batched_frag_graph_list_comp2,
                                                                            fetched_data.all_batched_motif_graph_list_comp1, fetched_data.all_batched_motif_graph_list_comp2,
                                                                            fetched_data.all_Tb_comp1,
                                                                            fetched_data.all_Tc_comp1,
                                                                            fetched_data.all_Tb_comp2,
                                                                            fetched_data.all_Tc_comp2,
                                                                            fetched_data.all_x1,
                                                                            fetched_data.all_T,fetched_data.all_smiles_list_comp1,fetched_data.all_smiles_list_comp2,n_paramT)
        
        modelY = early_stoppingY.load_checkpoint(modelY)
        _, epoch_train_Y_metrics = evaluate_frag_VLE(modelY, scaling_y1,fetched_data.train_iter,
                                                                            fetched_data.train_batched_origin_graph_list_comp1,
                                                                            fetched_data.train_batched_origin_graph_list_comp2,
                                                                            fetched_data.train_batched_frag_graph_list_comp1,
                                                                            fetched_data.train_batched_frag_graph_list_comp2,
                                                                            fetched_data.train_batched_motif_graph_list_comp1,
                                                                            fetched_data.train_batched_motif_graph_list_comp2,
                                                                            fetched_data.train_Tb_comp1,
                                                                            fetched_data.train_Tc_comp1,
                                                                            fetched_data.train_Tb_comp2,
                                                                            fetched_data.train_Tc_comp2,
                                                                            fetched_data.train_x1,
                                                                            fetched_data.train_y1,fetched_data.train_smiles_list_comp1,fetched_data.train_smiles_list_comp2,n_paramY)
        _, epoch_val_Y_metrics = evaluate_frag_VLE(modelY,scaling_y1,fetched_data.val_iter, 
                                                                            fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                            fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                            fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                            fetched_data.val_Tb_comp1,
                                                                            fetched_data.val_Tc_comp1,
                                                                            fetched_data.val_Tb_comp2,
                                                                            fetched_data.val_Tc_comp2,
                                                                            fetched_data.val_x1,
                                                                            fetched_data.val_y1,fetched_data.val_smiles_list_comp1,fetched_data.val_smiles_list_comp2,n_paramY)
        _, epoch_test_Y_metrics = evaluate_frag_VLE(modelY,scaling_y1,fetched_data.test_iter, 
                                                                            fetched_data.test_batched_origin_graph_list_comp1, fetched_data.test_batched_origin_graph_list_comp2,
                                                                            fetched_data.test_batched_frag_graph_list_comp1, fetched_data.test_batched_frag_graph_list_comp2,
                                                                            fetched_data.test_batched_motif_graph_list_comp1, fetched_data.test_batched_motif_graph_list_comp2,
                                                                            fetched_data.test_Tb_comp1,
                                                                            fetched_data.test_Tc_comp1,
                                                                            fetched_data.test_Tb_comp2,
                                                                            fetched_data.test_Tc_comp2,
                                                                            fetched_data.test_x1,
                                                                            fetched_data.test_y1,fetched_data.test_smiles_list_comp1,fetched_data.test_smiles_list_comp2,n_paramY)
        _, epoch_raw_Y_metrics = evaluate_frag_VLE(modelY,scaling_y1,fetched_data.all_iter, 
                                                                            fetched_data.all_batched_origin_graph_list_comp1, fetched_data.all_batched_origin_graph_list_comp2,
                                                                            fetched_data.all_batched_frag_graph_list_comp1, fetched_data.all_batched_frag_graph_list_comp2,
                                                                            fetched_data.all_batched_motif_graph_list_comp1, fetched_data.all_batched_motif_graph_list_comp2,
                                                                            fetched_data.all_Tb_comp1,
                                                                            fetched_data.all_Tc_comp1,
                                                                            fetched_data.all_Tb_comp2,
                                                                            fetched_data.all_Tc_comp2,
                                                                            fetched_data.all_x1,
                                                                            fetched_data.all_y1,fetched_data.all_smiles_list_comp1,fetched_data.all_smiles_list_comp2,n_paramY)
        wr.loc[i] = [seed, epoch_train_T_metrics.R2, epoch_val_T_metrics.R2,
                epoch_test_T_metrics.R2, epoch_raw_T_metrics.R2,
                epoch_train_Y_metrics.R2, epoch_val_Y_metrics.R2,
                epoch_test_Y_metrics.R2, epoch_raw_Y_metrics.R2,
                epoch_train_T_metrics.MAE, epoch_val_T_metrics.MAE, epoch_test_T_metrics.MAE,  epoch_raw_T_metrics.MAE,
                epoch_train_T_metrics.RMSE, epoch_val_T_metrics.RMSE, epoch_test_T_metrics.RMSE,  epoch_raw_T_metrics.RMSE,
                epoch_train_Y_metrics.MAE, epoch_val_Y_metrics.MAE, epoch_test_Y_metrics.MAE,  epoch_raw_Y_metrics.MAE,
                epoch_train_Y_metrics.RMSE, epoch_val_Y_metrics.RMSE, epoch_test_Y_metrics.RMSE,  epoch_raw_Y_metrics.RMSE]
    wr.to_csv(save_file_path)
            
            
            
            
            

