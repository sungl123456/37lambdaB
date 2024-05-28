from dataload.dataloading import import_dataset
from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.junctiontree_ring_encoder import frag
from utils.junctiontree_encoder import JT_SubGraph
from dataload.Azeotrope_dataset import Azeotrope_Dataset
from dataload.pair_dataset import pair_Dataset
from dataload.VLE_dataset import VLE_Dataset
from utils.mol2graph import smiles_2_bigraph
# from utils.splitter import Splitter
from utils.VLE_splitter import Azeotrope_Splitter, pair_Splitter,VLE_Splitter,VLE_Splitter2
from src.dgltools import collate_fraggraphs_VLE2, collate_fraggraphs_pair
from utils.Earlystopping import EarlyStopping
from utils.count_parameters import count_parameters
from utils.VLE_piplines import Azeotrope_PreFetch,train_epoch_frag_VLE,predict_Tb_Tc, evaluate_frag_VLE,evaluate_frag_VLE_attention_value
from utils.piplines import train_epoch, evaluate, train_epoch_frag, evaluate_frag, PreFetch, evaluate_frag_descriptors, evaluate_frag_attention

import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import math

from networks.DMPNN import DMPNNNet
from networks.MPNN import MPNNNet
from networks.AttentiveFP import AttentiveFPNet
from networks.FraGAT import NewFraGATNet
from networks.AGC import AGCNet
from networks.AGC_VLE2 import AGCNetVLE2

from dataload.model_library import save_model, load_model, load_optimal_model

params = {}
net_params = {}

params['init_lr'] = 10 ** -2
params['min_lr'] = 1e-9
params['weight_decay'] = 10 ** -6
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
net_params['layers'] = 1
net_params['residual'] = False
net_params['batch_norm'] = False
net_params['layer_norm'] = False
net_params['device'] = 'cpu'
splitting_seed = [3386,1141]
dataset_list = ['zeotrope_VLE','azeotrope_VLE']
path_l = ['Ensembles/Tb_VLE_normalized/MG_normalized','Ensembles/Tc_VLE_normalized/MG_normalized']
name_l = ['Ensemble_0_Tb_VLE_normalized_AGCNet','Ensemble_0_Tc_VLE_normalized_AGCNet']

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
    comp1_Tb = pd.DataFrame(comp1_Tb)
    comp2_Tb = pd.DataFrame(comp2_Tb)
    comp1_Tc = pd.DataFrame(comp1_Tc)
    comp2_Tc = pd.DataFrame(comp2_Tc)
    df['comp1_Tb'] = comp1_Tb
    df['comp2_Tb'] = comp2_Tb
    df['comp1_Tc'] = comp1_Tc
    df['comp2_Tc'] = comp2_Tc
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
                                    load = True,
                                    fragmentation=fragmentation)
    
    split = VLE_Splitter2(allset)

    train_dataset, val_dataset, test_dataset, all_dataset = split.Random_Splitter(comp, seed = 100, frac_train = 0.8, frac_val = 0.9)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fraggraphs_VLE2, batch_size=len(train_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    val_loader = DataLoader(val_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(val_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    test_loader = DataLoader(test_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(test_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    raw_loader = DataLoader(all_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(all_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))

    fetched_data = Azeotrope_PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=4)
    init_seed_list = [534]
    torch.manual_seed(init_seed_list[0])

    model = AGCNetVLE2(net_params).to(device='cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr = params['init_lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                                patience=params['lr_schedule_patience'], verbose=False)
    t0 = time.time()
    per_epoch_time = []
    early_stopping = EarlyStopping(patience=params['earlystopping_patience'], path='checkpoint_ensemble_' + params['Dataset'] + '_AGCNet_VLE2' + '_MG_' + str(0) + '.pt')

    with tqdm(range(params['max_epoch'])) as t:
        n_param = count_parameters(model)
        for epoch in t:
            t.set_description('Epoch %d' % epoch)
            start = time.time()
            model, epoch_train_loss, epoch_train_metrics  = train_epoch_frag_VLE(model, optimizer, scaling_y1,
                                                                                        fetched_data.all_iter,
                                                                                        fetched_data.all_batched_origin_graph_list_comp1,
                                                                                        fetched_data.all_batched_origin_graph_list_comp2,
                                                                                        fetched_data.all_batched_frag_graph_list_comp1,
                                                                                        fetched_data.all_batched_frag_graph_list_comp2,
                                                                                        fetched_data.all_batched_motif_graph_list_comp1,
                                                                                        fetched_data.all_batched_motif_graph_list_comp2,
                                                                                        fetched_data.all_Tb_comp1,
                                                                                        fetched_data.all_Tc_comp1,
                                                                                        fetched_data.all_Tb_comp2,
                                                                                        fetched_data.all_Tc_comp2,
                                                                                        fetched_data.all_x1,
                                                                                        fetched_data.all_y1,
                                                                                        n_param)
            

            epoch_val_loss, epoch_val_metrics = evaluate_frag_VLE(model, scaling_T, fetched_data.val_iter, 
                                                                            fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                            fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                            fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                            fetched_data.val_Tb_comp1,
                                                                            fetched_data.val_Tc_comp1,
                                                                            fetched_data.val_Tb_comp2,
                                                                            fetched_data.val_Tc_comp2,
                                                                            fetched_data.val_x1,
                                                                            fetched_data.val_y1,
                                                                            n_param)
            
            t.set_postfix({'time': time.time() - start, 'lr': optimizer.param_groups[0]['lr'],
                'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss, 
                'train_R2': epoch_train_metrics.R2, 'val_R2': epoch_val_metrics.R2})
            per_epoch_time.append(time.time() - start)

            scheduler.step(epoch_val_loss)
            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print('\n! LR equal to min LR set.')
                break

            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                break

        model = early_stopping.load_checkpoint(model)

        _, epoch_train_metrics = evaluate_frag_VLE(model, scaling_T,fetched_data.train_iter,
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
                                                                            fetched_data.train_y1,n_param)
        _, epoch_val_metrics = evaluate_frag_VLE(model,scaling_T,fetched_data.val_iter, 
                                                                            fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                            fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                            fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                            fetched_data.val_Tb_comp1,
                                                                            fetched_data.val_Tc_comp1,
                                                                            fetched_data.val_Tb_comp2,
                                                                            fetched_data.val_Tc_comp2,
                                                                            fetched_data.val_x1,
                                                                            fetched_data.val_y1,n_param)
        _, epoch_test_metrics = evaluate_frag_VLE(model,scaling_T,fetched_data.test_iter, 
                                                                            fetched_data.test_batched_origin_graph_list_comp1, fetched_data.test_batched_origin_graph_list_comp2,
                                                                            fetched_data.test_batched_frag_graph_list_comp1, fetched_data.test_batched_frag_graph_list_comp2,
                                                                            fetched_data.test_batched_motif_graph_list_comp1, fetched_data.test_batched_motif_graph_list_comp2,
                                                                            fetched_data.test_Tb_comp1,
                                                                            fetched_data.test_Tc_comp1,
                                                                            fetched_data.test_Tb_comp2,
                                                                            fetched_data.test_Tc_comp2,
                                                                            fetched_data.test_x1,
                                                                            fetched_data.test_y1,n_param)
        _, epoch_raw_metrics, all_predict, all_target,\
        attention_comp1_list, attention_comp2_list, \
        smiles_list_comp1, smiles_list_comp2 = evaluate_frag_VLE_attention_value(model,scaling_T,fetched_data.all_iter, 
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
                                                                            fetched_data.all_smiles_list_comp2,n_param)


        path = '/Ensembles/'+ params['Dataset'] + '/MG_Y_ensemble'
        name = '{}_{}_{}'.format('Ensemble_0', params['Dataset'], 'AGCNetVLE2')
        results = pd.Series({'init_seed': init_seed_list[0],'seed': splitting_seed[j], 'train_R2': epoch_train_metrics.R2, 'val_R2': epoch_val_metrics.R2,
                        'test_R2': epoch_test_metrics.R2, 'all_R2': epoch_raw_metrics.R2,
                        'train_MAE': epoch_train_metrics.MAE, 'val_MAE': epoch_val_metrics.MAE,
                        'test_MAE': epoch_test_metrics.MAE, 'all_MAE': epoch_raw_metrics.MAE,
                        'train_RMSE': epoch_train_metrics.RMSE, 'val_RMSE': epoch_val_metrics.RMSE,
                        'test_RMSE': epoch_test_metrics.RMSE, 'all_RMSE': epoch_raw_metrics.RMSE, 'train_SSE': epoch_train_metrics.SSE,
                        'val_SSE': epoch_val_metrics.SSE, 'test_SSE': epoch_test_metrics.SSE, 'all_SSE': epoch_raw_metrics.SSE,
                        'train_MAPE': epoch_train_metrics.MAPE, 'val_MAPE': epoch_val_metrics.MAPE, 'test_MAPE': epoch_test_metrics.MAPE,
                        'all_MAPE': epoch_raw_metrics.MAPE})
        comments = ''
        save_model(model, path, name, params, net_params, results, comments)

        df_value = pd.DataFrame({'SMILES_comp1': smiles_list_comp1[0], 'SMILES_comp2': smiles_list_comp2[0],
                            'Target': all_target.numpy().flatten().tolist(),
                            'Predict': all_predict.numpy().flatten().tolist()})
        df_attention = pd.DataFrame({'SMILES_comp1': smiles_list_comp1[0], 'SMILES_comp2': smiles_list_comp2[0],
                                'attention_comp1':np.array([v.tolist() for v in attention_comp1_list],dtype = list),
                                'attention_comp2':np.array([v.tolist() for v in attention_comp2_list],dtype = list)})
        df_results = pd.concat([df_value, df_attention])
        # op_idx, init_seed, seed, params, net_params, model = load_optimal_model(path, name)
        save_file_path = os.path.join('./library/' + path, '{}_{}_{}_{}'.format(name, dataset_list[j], splitting_seed, time.strftime('%Y-%m-%d-%H-%M')) + '.csv')
        df_results.to_csv(save_file_path, index=False)

