# this file is used for developing a azeotropic classification model with suitable random seed
from dataload.dataloading import import_dataset
from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.junctiontree_encoder import JT_SubGraph
from dataload.Azeotrope_dataset import Azeotrope_Dataset
from dataload.pair_dataset import pair_Dataset
from utils.mol2graph import smiles_2_bigraph
from utils.VLE_splitter import pair_Splitter3
from src.dgltools import collate_fraggraphs_VLE,collate_fraggraphs_pair
from utils.count_parameters import count_parameters
from utils.VLE_piplines import Azeotrope_PreFetch, predict_Tb_Tc,train_epoch_frag_clf,evaluate_frag_clf
from utils.piplines import PreFetch
from utils.Set_Seed_Reproducibility import set_seed

import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import math

from networks.AGC_clf import AGCNetCLF

from dataload.model_library import save_model, load_model
params = {}
net_params = {}
params['init_lr'] = 10 ** -3  
params['min_lr'] = 10 ** -8
params['weight_decay'] = 0
params['lr_reduce_factor'] = 0.75
params['lr_schedule_patience'] = 15
params['earlystopping_patience'] = 40  
params['max_epoch'] = 300          

net_params['num_atom_type'] = 36
net_params['num_bond_type'] = 12
net_params['hidden_dim'] = 32   
net_params['dropout'] = 0       
net_params['num_heads'] = 2
net_params['depth'] = 2
net_params['layers'] = 3
net_params['residual'] = False
net_params['batch_norm'] = False
net_params['layer_norm'] = True
net_params['device'] = 'cpu'
splitting_seed = 915
dataset_list = ['azeotrope_classification_cleaned']
path_l = ['Tb_JCIM_normalized/all','Tc_JCIM_normalized/all']
name_l = ['Ensemble_0_Tb_JCIM_normalized_AGCNet','Ensemble_0_Tc_JCIM_normalized_AGCNet']

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
init_seed_list = [random.randint(0,1000) for i in range(100)]

for i in range(50):
    torch.manual_seed(init_seed_list[i])
    train_dataset, val_dataset, test_dataset, all_dataset = split.Random_Splitter(comp, seed = splitting_seed, frac_train = 0.8, frac_val = 0.9)
    train_loader = DataLoader(train_dataset, collate_fn=collate_fraggraphs_VLE, batch_size=len(train_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    val_loader = DataLoader(val_dataset,collate_fn=collate_fraggraphs_VLE, batch_size=len(val_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    test_loader = DataLoader(test_dataset,collate_fn=collate_fraggraphs_VLE, batch_size=len(test_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    raw_loader = DataLoader(all_dataset,collate_fn=collate_fraggraphs_VLE, batch_size=len(all_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))

    fetched_data = Azeotrope_PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=3)

    model = AGCNetCLF(net_params).to(device='cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr = params['init_lr'],weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                                patience=params['lr_schedule_patience'])

    t0 = time.time()
    per_epoch_time = []


    with tqdm(range(params['max_epoch'])) as t:
        n_param = count_parameters(model)
        for epoch in t:
            t.set_description('Epoch %d' % epoch)
            start = time.time()
            model, epoch_train_loss, epoch_train_metrics  = train_epoch_frag_clf(model, optimizer, fetched_data.train_iter,
                                                                fetched_data.train_batched_origin_graph_list_comp1, fetched_data.train_batched_origin_graph_list_comp2,
                                                                fetched_data.train_batched_frag_graph_list_comp1, fetched_data.train_batched_frag_graph_list_comp2,
                                                                fetched_data.train_batched_motif_graph_list_comp1, fetched_data.train_batched_motif_graph_list_comp2,
                                                                fetched_data.train_Tb_comp1,
                                                                fetched_data.train_Tc_comp1,
                                                                fetched_data.train_Tb_comp2,
                                                                fetched_data.train_Tc_comp2,
                                                                fetched_data.train_targets_list)

            epoch_val_loss, epoch_val_metrics = evaluate_frag_clf(model, fetched_data.val_iter, 
                                                                fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                fetched_data.val_Tb_comp1,
                                                                fetched_data.val_Tc_comp1,
                                                                fetched_data.val_Tb_comp2,
                                                                fetched_data.val_Tc_comp2,
                                                                fetched_data.val_targets_list)

            epoch_test_loss, epoch_test_metrics = evaluate_frag_clf(model, fetched_data.test_iter, 
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
                                                                    fetched_data.test_targets_list)
            
            t.set_postfix({'time': time.time() - start, 'lr': optimizer.param_groups[0]['lr'],
                'train_score': epoch_train_metrics.score, 'val_score': epoch_val_metrics.score, 'test_score':epoch_test_metrics.score})
            per_epoch_time.append(time.time() - start)

            scheduler.step(epoch_val_loss)
            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print('\n! LR equal to min LR set.')
                break

    _, epoch_train_metrics = evaluate_frag_clf(model, fetched_data.train_iter, 
                                                                    fetched_data.train_batched_origin_graph_list_comp1, fetched_data.train_batched_origin_graph_list_comp2,
                                                                    fetched_data.train_batched_frag_graph_list_comp1, fetched_data.train_batched_frag_graph_list_comp2,
                                                                    fetched_data.train_batched_motif_graph_list_comp1, fetched_data.train_batched_motif_graph_list_comp2,
                                                                    fetched_data.train_Tb_comp1,
                                                                    fetched_data.train_Tc_comp1,
                                                                    fetched_data.train_Tb_comp2,
                                                                    fetched_data.train_Tc_comp2,
                                                                    fetched_data.train_targets_list)

    _, epoch_val_metrics = evaluate_frag_clf(model,fetched_data.val_iter, 
                                                                    fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                    fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                    fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                    fetched_data.val_Tb_comp1,
                                                                    fetched_data.val_Tc_comp1,
                                                                    fetched_data.val_Tb_comp2,
                                                                    fetched_data.val_Tc_comp2,
                                                                    fetched_data.val_targets_list)

    _, epoch_test_metrics, test_scores, targets_test, result_test, test_smiles_comp1, test_smiles_comp2 = evaluate_frag_clf(model,fetched_data.test_iter, 
                                                                    fetched_data.test_batched_origin_graph_list_comp1, fetched_data.test_batched_origin_graph_list_comp2,
                                                                    fetched_data.test_batched_frag_graph_list_comp1, fetched_data.test_batched_frag_graph_list_comp2,
                                                                    fetched_data.test_batched_motif_graph_list_comp1, fetched_data.test_batched_motif_graph_list_comp2,
                                                                    fetched_data.test_Tb_comp1,
                                                                    fetched_data.test_Tc_comp1,
                                                                    fetched_data.test_Tb_comp2,
                                                                    fetched_data.test_Tc_comp2,
                                                                    fetched_data.test_targets_list,
                                                                    fetched_data.test_smiles_list_comp1,
                                                                    fetched_data.test_smiles_list_comp2,flag=True)

    _, epoch_raw_metrics,all_scores, targets_all, result_all, all_smiles_comp1, all_smiles_comp2 = evaluate_frag_clf(model, fetched_data.all_iter,
                                                                    fetched_data.all_batched_origin_graph_list_comp1, fetched_data.all_batched_origin_graph_list_comp2,
                                                                    fetched_data.all_batched_frag_graph_list_comp1, fetched_data.all_batched_frag_graph_list_comp2,
                                                                    fetched_data.all_batched_motif_graph_list_comp1, fetched_data.all_batched_motif_graph_list_comp2,
                                                                    fetched_data.all_Tb_comp1,
                                                                    fetched_data.all_Tc_comp1,
                                                                    fetched_data.all_Tb_comp2,
                                                                    fetched_data.all_Tc_comp2,
                                                                    fetched_data.all_targets_list,
                                                                    fetched_data.all_smiles_list_comp1,
                                                                    fetched_data.all_smiles_list_comp2,flag=True)

    path = dataset_list[0]
    name = '{}_{}_{}'.format('Ensemble_0', params['Dataset'], 'AGCNetCLF')
    results = pd.Series({'init_seed': init_seed_list[i], 'seed': splitting_seed,
                        'train_score': epoch_train_metrics.score, 'val_score': epoch_val_metrics.score,
                        'test_score': epoch_test_metrics.score, 'all_score': epoch_raw_metrics.score,
                        'train_confusion_matirx': epoch_train_metrics.confusion,
                        'val_confusion_matrix': epoch_val_metrics.confusion,
                        'test_confusion_matrix': epoch_test_metrics.confusion,
                        'all_confusion_matrix': epoch_raw_metrics.confusion})

    comments = ''
    save_model(model, path, name, params, net_params, results, comments)