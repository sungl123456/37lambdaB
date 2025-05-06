
from dataload.dataloading import import_dataset
from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.junctiontree_encoder import JT_SubGraph
from dataload.Azeotrope_dataset import Azeotrope_Dataset
from dataload.pair_dataset import pair_Dataset
from utils.mol2graph import smiles_2_bigraph
from utils.VLE_splitter import pair_Splitter3
from src.dgltools import collate_fraggraphs_VLE, collate_fraggraphs_pair
from utils.Earlystopping import EarlyStopping
from utils.count_parameters import count_parameters
from utils.VLE_piplines import Azeotrope_PreFetch,train_epoch_frag_clf,predict_Tb_Tc, evaluate_frag_clf
from utils.Set_Seed_Reproducibility import set_seed

import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd



from networks.AGC_VLE import AGCNetVLE

from dataload.model_library import load_model

params = {}
net_params = {}

params['init_lr'] = 10 ** -2
params['min_lr'] = 1e-9
params['weight_decay'] = 0
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
dataset_list = ['azeotrope_classification']
path_l = ['Tb_JCIM_normalized/all','Tc_JCIM_normalized/all']
name_l = ['Ensemble_0_Tb_JCIM_normalized_AGCNet','Ensemble_0_Tc_JCIM_normalized_AGCNet']

for j in range(len(dataset_list)):
    params['Dataset'] = dataset_list[j]
    net_params['Dataset'] = dataset_list[j]
    df = import_dataset(params)

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

    rows = []
    file_path = os.path.realpath('./output')
    if not os.path.exists(file_path):
         os.mkdir(file_path)
    save_file_path = os.path.join(file_path,'{}_{}_{}'.format(params['Dataset'], 'AGCNet_Pair', time.strftime('%Y-%m-%d-%H-%M'))+'.csv')
    wr = pd.DataFrame(columns=['seed', 'train_score', 'val_score', 'test_score', 'all_score'])

    for i in range(50):
        seed = np.random.randint(1,5000)
        set_seed(seed = 1000)
        train_dataset, val_dataset, test_dataset, all_dataset = split.Random_Splitter(comp, seed = seed, frac_train = 0.8, frac_val = 0.9)

        train_loader = DataLoader(train_dataset, collate_fn=collate_fraggraphs_VLE, batch_size=len(train_dataset), shuffle=False)
        val_loader = DataLoader(val_dataset,collate_fn=collate_fraggraphs_VLE, batch_size=len(val_dataset), shuffle=False)
        test_loader = DataLoader(test_dataset,collate_fn=collate_fraggraphs_VLE, batch_size=len(test_dataset), shuffle=False)
        raw_loader = DataLoader(all_dataset,collate_fn=collate_fraggraphs_VLE, batch_size=len(all_dataset), shuffle=False)

        fetched_data = Azeotrope_PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=3)

        model = AGCNetVLE(net_params).to(device='cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr = params['init_lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                                    patience=params['lr_schedule_patience'])

        t0 = time.time()
        per_epoch_time = []
        early_stopping = EarlyStopping(patience=params['earlystopping_patience'], path='./checkpoint/checkpoint_ensemble_' + params['Dataset'] + '_AGCNet_pair' + '.pt')
        
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
                    # 'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss, 'test_loss': epoch_test_loss,
                    'train_score': epoch_train_metrics.score, 'val_score': epoch_val_metrics.score, 'test_score':epoch_test_metrics.score})
                per_epoch_time.append(time.time() - start)

                scheduler.step(epoch_val_loss)
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print('\n! LR equal to min LR set.')
                    break

                early_stopping(epoch_val_loss, model)
                if early_stopping.early_stop:
                    break

        model = early_stopping.load_checkpoint(model)
        _, epoch_train_metrics = evaluate_frag_clf(model, fetched_data.train_iter,
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
                                                        fetched_data.train_targets_list)
        
        _, epoch_val_metrics = evaluate_frag_clf(model, fetched_data.val_iter, 
                                                        fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                        fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                        fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                        fetched_data.val_Tb_comp1,
                                                        fetched_data.val_Tc_comp1,
                                                        fetched_data.val_Tb_comp2,
                                                        fetched_data.val_Tc_comp2,
                                                        fetched_data.val_targets_list)
        _, epoch_test_metrics = evaluate_frag_clf(model, fetched_data.test_iter, 
                                                        fetched_data.test_batched_origin_graph_list_comp1, fetched_data.test_batched_origin_graph_list_comp2,
                                                        fetched_data.test_batched_frag_graph_list_comp1, fetched_data.test_batched_frag_graph_list_comp2,
                                                        fetched_data.test_batched_motif_graph_list_comp1, fetched_data.test_batched_motif_graph_list_comp2,
                                                        fetched_data.test_Tb_comp1,
                                                        fetched_data.test_Tc_comp1,
                                                        fetched_data.test_Tb_comp2,
                                                        fetched_data.test_Tc_comp2,
                                                        fetched_data.test_targets_list)
        _, epoch_raw_metrics = evaluate_frag_clf(model, fetched_data.all_iter, 
                                                        fetched_data.all_batched_origin_graph_list_comp1, fetched_data.all_batched_origin_graph_list_comp2,
                                                        fetched_data.all_batched_frag_graph_list_comp1, fetched_data.all_batched_frag_graph_list_comp2,
                                                        fetched_data.all_batched_motif_graph_list_comp1, fetched_data.all_batched_motif_graph_list_comp2,
                                                        fetched_data.all_Tb_comp1,
                                                        fetched_data.all_Tc_comp1,
                                                        fetched_data.all_Tb_comp2,
                                                        fetched_data.all_Tc_comp2,
                                                        fetched_data.all_targets_list)

        wr.loc[i] = [seed, epoch_train_metrics.score, epoch_val_metrics.score, epoch_test_metrics.score, epoch_raw_metrics.score]
    wr.to_csv(save_file_path)
            
            
            
            
            

