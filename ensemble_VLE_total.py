# this file is built for developing a VLE prediction model with suitable random seed
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
from utils.count_parameters import count_parameters
from utils.VLE_piplines import Azeotrope_PreFetch,train_epoch_frag_VLE, predict_Tb_Tc, evaluate_frag_VLE
from utils.Set_Seed_Reproducibility import set_seed

import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from networks.AGC_VLE import AGCNetVLE
from dataload.model_library import save_model, load_model

params = {}
net_params = {}

params['init_lr'] = 10 ** -3
params['min_lr'] = 1e-8
params['weight_decay'] = 0
params['lr_reduce_factor'] = 0.75
params['lr_schedule_patience'] = 30
params['earlystopping_patience'] = 50
params['max_epoch'] = 500

net_params['num_atom_type'] = 36
net_params['num_bond_type'] = 12
net_params['hidden_dim'] = 128
net_params['num_heads'] = 2
net_params['dropout'] = 0
net_params['depth'] = 2
net_params['layers'] = 1
net_params['residual'] = False
net_params['batch_norm'] = False
net_params['layer_norm'] = True
net_params['device'] = 'cpu'

splitting_seed = [3630]
dataset_list = ['VLE_total_cleaned']
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
    
    set_seed(seed =1000)
    init_seed_list = [random.randint(0,1000) for i in range(100)]
    for i in range(50):
        torch.manual_seed(init_seed_list[i])
        train_dataset, val_dataset, test_dataset, all_dataset = split.Random_Splitter(comp, seed = splitting_seed[j], frac_train = 0.8, frac_val = 0.9)

        train_loader = DataLoader(train_dataset, collate_fn=collate_fraggraphs_VLE2, batch_size=len(train_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
        val_loader = DataLoader(val_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(val_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
        test_loader = DataLoader(test_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(test_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
        raw_loader = DataLoader(all_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(all_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))

        fetched_data = Azeotrope_PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=4)


        model = AGCNetVLE(net_params).to(device='cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr = params['init_lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                                    patience=params['lr_schedule_patience'])
        
        t0 = time.time()
        per_epoch_time = []
        
        with tqdm(range(params['max_epoch'])) as t:
            n_param = count_parameters(model)
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                model, epoch_train_loss, epoch_train_metrics_T, epoch_train_metrics_Y  = train_epoch_frag_VLE(model, optimizer, scaling_T, scaling_y1,
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
                                                                                            fetched_data.train_y1,
                                                                                            n_param)
                

                epoch_val_loss, epoch_val_metrics_T,epoch_val_metrics_Y = evaluate_frag_VLE(model, scaling_T,scaling_y1,
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
                                                                    fetched_data.val_y1,
                                                                    fetched_data.val_smiles_list_comp1,
                                                                    fetched_data.val_smiles_list_comp2,
                                                                    n_param)

                epoch_test_loss, epoch_test_metrics_T,epoch_test_metrics_Y = evaluate_frag_VLE(model, scaling_T,scaling_y1,
                                                                        fetched_data.test_iter, 
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
                                                                        fetched_data.test_smiles_list_comp2,
                                                                        n_param)
                
                t.set_postfix({'time': time.time() - start, 'lr': optimizer.param_groups[0]['lr'],
                    'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss, 
                    'R2_T': epoch_train_metrics_T.R2, 'R2_Y': epoch_train_metrics_Y.R2})

                scheduler.step(epoch_val_loss)
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print('\n! LR equal to min LR set.')
                    break

        _, epoch_train_metrics_T, epoch_train_metrics_Y = evaluate_frag_VLE(model, scaling_T,scaling_y1,fetched_data.train_iter,
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
                                                                            fetched_data.train_y1,
                                                                            fetched_data.train_smiles_list_comp1,
                                                                            fetched_data.train_smiles_list_comp2,n_param)
        _, epoch_val_metrics_T,epoch_val_metrics_Y = evaluate_frag_VLE(model,scaling_T,scaling_y1,fetched_data.val_iter, 
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
        _, epoch_test_metrics_T, epoch_test_metrics_Y, predict_test_T, target_test_T, predict_test_Y, target_test_Y,test_smiles_comp1, test_smiles_comp2 = \
                                                                            evaluate_frag_VLE(model,scaling_T,scaling_y1,fetched_data.test_iter, 
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
                                                                            fetched_data.test_smiles_list_comp2,n_param,flag = True)
        _, epoch_raw_metrics_T, epoch_raw_metrics_Y,predict_all_T, target_all_T, predict_all_Y, target_all_Y,all_smiles_comp1, all_smiles_comp2 = evaluate_frag_VLE(model,scaling_T,scaling_y1,fetched_data.all_iter, 
                                                                            fetched_data.all_batched_origin_graph_list_comp1, fetched_data.all_batched_origin_graph_list_comp2,
                                                                            fetched_data.all_batched_frag_graph_list_comp1, fetched_data.all_batched_frag_graph_list_comp2,
                                                                            fetched_data.all_batched_motif_graph_list_comp1, fetched_data.all_batched_motif_graph_list_comp2,
                                                                            fetched_data.all_Tb_comp1,
                                                                            fetched_data.all_Tc_comp1,
                                                                            fetched_data.all_Tb_comp2,
                                                                            fetched_data.all_Tc_comp2,
                                                                            fetched_data.all_x1,
                                                                            fetched_data.all_T,
                                                                            fetched_data.all_y1,
                                                                            fetched_data.all_smiles_list_comp1,
                                                                            fetched_data.all_smiles_list_comp2,n_param, flag = True)


        path = params['Dataset']
        name = '{}_{}_{}'.format('Ensemble_0', params['Dataset'], 'AGCNetVLE')
        results = pd.Series({'init_seed': init_seed_list[0],'seed': splitting_seed[j], 
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
        save_model(model, path, name, params, net_params, results, comments)
        df_test = pd.DataFrame({'SMILES_comp1': test_smiles_comp1[0], 'SMILES_comp2': test_smiles_comp2[0], 'Tag': 'Test', 'Target_T': target_test_T.numpy().flatten().tolist(),
                                    'Predict_T': predict_test_T.numpy().flatten().tolist(), 'Target_Y':target_test_Y.numpy().flatten().tolist(),
                                    'Predict_Y': predict_test_Y.numpy().flatten().tolist()})

        save_file_path = os.path.join('./library/' + path, '{}_{}_{}_{}'.format(name, 'test', init_seed_list[i], time.strftime('%Y-%m-%d-%H-%M')) + '.csv')
        df_test.to_csv(save_file_path,index=False)

        df_raw = pd.DataFrame({'SMILES_comp1': all_smiles_comp1[0], 'SMILES_comp2': all_smiles_comp2[0], 'Tag': 'Test', 'Target_T': target_all_T.numpy().flatten().tolist(),
                                    'Predict_T': predict_all_T.numpy().flatten().tolist(), 'Target_Y':target_all_Y.numpy().flatten().tolist(),
                                    'Predict_Y': predict_all_Y.numpy().flatten().tolist()})

        save_raw_path = os.path.join('./library/' + path, '{}_{}_{}_{}'.format(name, 'all', init_seed_list[i], time.strftime('%Y-%m-%d-%H-%M')) + '.csv')
        df_raw.to_csv(save_raw_path,index=False)
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))




