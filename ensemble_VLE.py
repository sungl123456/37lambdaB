
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
from utils.VLE_piplines import Azeotrope_PreFetch,train_epoch_frag_VLE4,predict_Tb_Tc, evaluate_frag_VLE4
from utils.Set_Seed_Reproducibility import set_seed

import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from networks.AGC_VLE5 import AGCNetVLE5
from networks.AGC_VLE6 import AGCNetVLE6
from dataload.model_library import save_model, load_model

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
    df['comp1_Tb'] = pd.DataFrame(comp1_Tb)
    df['comp2_Tb'] = pd.DataFrame(comp2_Tb)
    df['comp1_Tc'] = pd.DataFrame(comp1_Tc)
    df['comp2_Tc'] = pd.DataFrame(comp2_Tc)
    del df['value']

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
                                    load = False,
                                    fragmentation=fragmentation)
    
    split = VLE_Splitter2(allset)
    
    set_seed(seed =1000)
    init_seed_list = [random.randint(0,1000) for i in range(100)]
    for j in range(10):
        torch.manual_seed(init_seed_list[j])
        train_dataset, val_dataset, test_dataset, all_dataset = split.Random_Splitter(comp, seed = splitting_seed[j], frac_train = 0.8, frac_val = 0.9)
    
        train_loader = DataLoader(train_dataset, collate_fn=collate_fraggraphs_VLE2, batch_size=len(train_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
        val_loader = DataLoader(val_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(val_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
        test_loader = DataLoader(test_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(test_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
        raw_loader = DataLoader(all_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(all_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    
        fetched_data = Azeotrope_PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=4)
        
    
        model_T = AGCNetVLE5(net_params).to(device='cpu')
        optimizer_T = torch.optim.Adam(model_T.parameters(), lr = params['init_lr'])
        scheduler_T = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_T, mode='min', factor=params['lr_reduce_factor'],
                                                                    patience=params['lr_schedule_patience'])
        
        model_Y = AGCNetVLE6(net_params).to(device='cpu')
        optimizer_Y = torch.optim.Adam(model_Y.parameters(), lr = params['init_lr'])
        scheduler_Y = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_Y, mode='min', factor=params['lr_reduce_factor'],
                                                                    patience=params['lr_schedule_patience'])
        t0 = time.time()
        per_epoch_time = []
        early_stopping_T = EarlyStopping(patience=params['earlystopping_patience'], path=checkpoint_path + '/checkpoint_ensemble_' + params['Dataset'] + '_AGCNet_VLE2' + '_T_' + str(0) + '.pt')
        early_stopping_Y = EarlyStopping(patience=params['earlystopping_patience'], path=checkpoint_path + '/checkpoint_ensemble_' + params['Dataset'] + '_AGCNet_VLE3' + '_Y_' + str(0) + '.pt')
        
        with tqdm(range(params['max_epoch'])) as t:
            n_param = count_parameters(model_T)
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                model_T, epoch_train_loss_T, epoch_train_metrics_T  = train_epoch_frag_VLE4(model_T, optimizer_T, scaling_T,
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
                                                                                            n_param,params['lam_T'])
                
    
                epoch_val_loss_T, epoch_val_metrics_T = evaluate_frag_VLE4(model_T, scaling_T,
                                                                    fetched_data.val_iter, 
                                                                    fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                    fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                    fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                    fetched_data.val_Tb_comp1,
                                                                    fetched_data.val_Tc_comp1,
                                                                    fetched_data.val_Tb_comp2,
                                                                    fetched_data.val_Tc_comp2,
                                                                    fetched_data.val_x1,
                                                                    fetched_data.val_T,
                                                                    fetched_data.val_smiles_list_comp1,
                                                                    fetched_data.val_smiles_list_comp2,
                                                                    n_param,params['lam_T'])
                
                t.set_postfix({'time': time.time() - start, 'lr': optimizer_T.param_groups[0]['lr'],
                    'train_loss': epoch_train_loss_T, 'val_loss': epoch_val_loss_T, 
                    'train_R2': epoch_train_metrics_T.R2, 'val_R2': epoch_val_metrics_T.R2})
                per_epoch_time.append(time.time() - start)
    
                scheduler_T.step(epoch_val_loss_T)
                if optimizer_T.param_groups[0]['lr'] < params['min_lr']:
                    print('\n! LR equal to min LR set.')
                    break
    
                early_stopping_T(epoch_val_loss_T, model_T)
                if early_stopping_T.early_stop:
                    break
        
        with tqdm(range(params['max_epoch'])) as t:
            n_param = count_parameters(model_Y)
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                model_Y, epoch_train_loss_Y, epoch_train_metrics_Y  = train_epoch_frag_VLE4(model_Y, optimizer_Y, scaling_y1,
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
                                                                                            n_param,params['lam_Y'])
                
    
                epoch_val_loss_Y, epoch_val_metrics_Y = evaluate_frag_VLE4(model_Y, scaling_y1,
                                                                    fetched_data.val_iter, 
                                                                    fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                    fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                    fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                    fetched_data.val_Tb_comp1,
                                                                    fetched_data.val_Tc_comp1,
                                                                    fetched_data.val_Tb_comp2,
                                                                    fetched_data.val_Tc_comp2,
                                                                    fetched_data.val_x1,
                                                                    fetched_data.val_y1,
                                                                    fetched_data.val_smiles_list_comp1,
                                                                    fetched_data.val_smiles_list_comp2,
                                                                    n_param,params['lam_Y'])
                
                t.set_postfix({'time': time.time() - start, 'lr': optimizer_Y.param_groups[0]['lr'],
                    'train_loss': epoch_train_loss_Y, 'val_loss': epoch_val_loss_Y, 
                    'train_R2': epoch_train_metrics_Y.R2, 'val_R2': epoch_val_metrics_Y.R2})
                per_epoch_time.append(time.time() - start)
    
                scheduler_Y.step(epoch_val_loss_Y)
                if optimizer_Y.param_groups[0]['lr'] < params['min_lr']:
                    print('\n! LR equal to min LR set.')
                    break
    
                early_stopping_Y(epoch_val_loss_Y, model_Y)
                if early_stopping_Y.early_stop:
                    break
            
    
        model_T = early_stopping_T.load_checkpoint(model_T)
    
        _, epoch_train_metrics_T = evaluate_frag_VLE4(model_T, scaling_T,fetched_data.train_iter,
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
                                                                            fetched_data.train_smiles_list_comp1,
                                                                            fetched_data.train_smiles_list_comp2,n_param,params['lam_T'])
        _, epoch_val_metrics_T = evaluate_frag_VLE4(model_T,scaling_T,fetched_data.val_iter, 
                                                                            fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                            fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                            fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                            fetched_data.val_Tb_comp1,
                                                                            fetched_data.val_Tc_comp1,
                                                                            fetched_data.val_Tb_comp2,
                                                                            fetched_data.val_Tc_comp2,
                                                                            fetched_data.val_x1,
                                                                            fetched_data.val_T,
                                                                            fetched_data.val_smiles_list_comp1,
                                                                            fetched_data.val_smiles_list_comp2,n_param,params['lam_T'])
        _, epoch_test_metrics_T, predict_test_T, target_test_T, test_T_smiles_comp1, test_T_smiles_comp2 = evaluate_frag_VLE4(model_T,scaling_T,fetched_data.test_iter, 
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
                                                                            fetched_data.test_smiles_list_comp2,n_param,params['lam_T'],flag = True)
        _, epoch_raw_metrics_T = evaluate_frag_VLE4(model_T,scaling_T,fetched_data.all_iter, 
                                                                            fetched_data.all_batched_origin_graph_list_comp1, fetched_data.all_batched_origin_graph_list_comp2,
                                                                            fetched_data.all_batched_frag_graph_list_comp1, fetched_data.all_batched_frag_graph_list_comp2,
                                                                            fetched_data.all_batched_motif_graph_list_comp1, fetched_data.all_batched_motif_graph_list_comp2,
                                                                            fetched_data.all_Tb_comp1,
                                                                            fetched_data.all_Tc_comp1,
                                                                            fetched_data.all_Tb_comp2,
                                                                            fetched_data.all_Tc_comp2,
                                                                            fetched_data.all_x1,
                                                                            fetched_data.all_T,
                                                                            fetched_data.all_smiles_list_comp1,
                                                                            fetched_data.all_smiles_list_comp2,n_param,params['lam_T'])
    
        model_Y = early_stopping_Y.load_checkpoint(model_Y)
    
        _, epoch_train_metrics_Y = evaluate_frag_VLE4(model_Y, scaling_y1,fetched_data.train_iter,
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
                                                                            fetched_data.train_smiles_list_comp1,
                                                                            fetched_data.train_smiles_list_comp2,n_param,params['lam_Y'])
        _, epoch_val_metrics_Y = evaluate_frag_VLE4(model_Y,scaling_y1,fetched_data.val_iter, 
                                                                            fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                            fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                            fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                            fetched_data.val_Tb_comp1,
                                                                            fetched_data.val_Tc_comp1,
                                                                            fetched_data.val_Tb_comp2,
                                                                            fetched_data.val_Tc_comp2,
                                                                            fetched_data.val_x1,
                                                                            fetched_data.val_y1,
                                                                            fetched_data.val_smiles_list_comp1,
                                                                            fetched_data.val_smiles_list_comp2,n_param,params['lam_Y'])
        _, epoch_test_metrics_Y, predict_test_Y, target_test_Y, test_Y_smiles_comp1, test_Y_smiles_comp2 = evaluate_frag_VLE4(model_Y,scaling_y1,fetched_data.test_iter, 
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
                                                                            fetched_data.test_smiles_list_comp2,n_param,params['lam_Y'], flag =True)
        _, epoch_raw_metrics_Y = evaluate_frag_VLE4(model_Y,scaling_y1,fetched_data.all_iter, 
                                                                            fetched_data.all_batched_origin_graph_list_comp1, fetched_data.all_batched_origin_graph_list_comp2,
                                                                            fetched_data.all_batched_frag_graph_list_comp1, fetched_data.all_batched_frag_graph_list_comp2,
                                                                            fetched_data.all_batched_motif_graph_list_comp1, fetched_data.all_batched_motif_graph_list_comp2,
                                                                            fetched_data.all_Tb_comp1,
                                                                            fetched_data.all_Tc_comp1,
                                                                            fetched_data.all_Tb_comp2,
                                                                            fetched_data.all_Tc_comp2,
                                                                            fetched_data.all_x1,
                                                                            fetched_data.all_y1,
                                                                            fetched_data.all_smiles_list_comp1,
                                                                            fetched_data.all_smiles_list_comp2,n_param,params['lam_Y'])
    
        path_T = params['Dataset'] + '/T'
        path_Y = params['Dataset'] + '/Y'
        name_T = '{}_{}_{}'.format('Ensemble_0', params['Dataset'], 'AGCNetVLE5')
        name_Y = '{}_{}_{}'.format('Ensemble_0', params['Dataset'], 'AGCNetVLE6')
        results = pd.Series({'init_seed': init_seed[j],'seed': splitting_seed[j], 
                                'train_R2_T': epoch_train_metrics_T.R2, 'val_R2_T': epoch_val_metrics_T.R2,
                            'test_R2_T': epoch_test_metrics_T.R2, 'all_R2_T': epoch_raw_metrics_T.R2,
                            'train_R2_Y': epoch_train_metrics_Y.R2, 'val_R2_Y': epoch_val_metrics_Y.R2,
                            'test_R2_Y': epoch_test_metrics_Y.R2, 'all_R2_Y': epoch_raw_metrics_Y.R2,
                            'train_MAE_T': epoch_train_metrics_T.MAE, 'val_MAE_T': epoch_val_metrics_T.MAE,
                            'test_MAE_T': epoch_test_metrics_T.MAE, 'all_MAE_T': epoch_raw_metrics_T.MAE,
                            'train_MAE_Y': epoch_train_metrics_Y.MAE, 'val_MAE_Y': epoch_val_metrics_Y.MAE,
                            'test_MAE_Y': epoch_test_metrics_Y.MAE, 'all_MAE_Y': epoch_raw_metrics_Y.MAE,
                            'train_MAE_T': epoch_train_metrics_T.MAE, 'val_MAE_T': epoch_val_metrics_T.MAE,
                            'test_MAE_T': epoch_test_metrics_T.MAE, 'all_MAE_T': epoch_raw_metrics_T.MAE,
                            'train_RMSE_T': epoch_train_metrics_T.RMSE, 'val_RMSE_T': epoch_val_metrics_T.RMSE,
                            'test_RMSE_T': epoch_test_metrics_T.RMSE, 'all_RMSE_T': epoch_raw_metrics_T.RMSE,
                            'train_RMSE_Y': epoch_train_metrics_Y.RMSE, 'val_RMSE_Y': epoch_val_metrics_Y.RMSE,
                            'test_RMSE_Y': epoch_test_metrics_Y.RMSE, 'all_RMSE_Y': epoch_raw_metrics_Y.RMSE})
    
        comments = ''
        save_model(model_T, path_T, name_T, params, net_params, results, comments)
        save_model(model_Y, path_Y, name_Y, params, net_params, results, comments)
    
        
        df_test_T = pd.DataFrame({'SMILES_comp1': test_T_smiles_comp1[0], 'SMILES_comp2': test_T_smiles_comp2[0], 'Tag': 'Test', 'Target': target_test_T.numpy().flatten().tolist(),
                                    'Predict': predict_test_T.numpy().flatten().tolist()})
        df_test_Y = pd.DataFrame({'SMILES_comp1': test_Y_smiles_comp1[0], 'SMILES_comp2': test_Y_smiles_comp2[0], 'Tag': 'Test', 'Target': target_test_Y.numpy().flatten().tolist(),
                                    'Predict': predict_test_Y.numpy().flatten().tolist()})
    
        
        save_file_path_T = os.path.join('./library/' + path_T, '{}_{}_{}_{}'.format(name_T, 'test_value', init_seed_list[i], time.strftime('%Y-%m-%d-%H-%M')) + '.csv')
        save_file_path_Y = os.path.join('./library/' + path_Y, '{}_{}_{}_{}'.format(name_Y, 'test_value', init_seed_list[i], time.strftime('%Y-%m-%d-%H-%M')) + '.csv')
        df_test_T.to_csv(save_file_path_T,index=False)
        df_test_Y.to_csv(save_file_path_Y,index=False)




