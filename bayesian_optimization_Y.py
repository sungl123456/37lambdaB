# 该文件的主要目的是对网络的超参数进行优化，以方便后续训练模型.
# The main purpose of this program is to perform Bayesian optimization on the networks's hyperparameters.

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
from utils.VLE_piplines import Azeotrope_PreFetch, train_epoch_frag_VLE4, predict_Tb_Tc, evaluate_frag_VLE4

import os
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


from networks.AGC_VLE6 import AGCNetVLE6
from utils.Set_Seed_Reproducibility import set_seed
from dataload.model_library import load_model

from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction

params = {}
net_params = {}

params['init_lr'] = 1e-3
params['min_lr'] = 1e-9
params['weight_decay'] = 0
params['lr_reduce_factor'] = 0.8
params['lr_schedule_patience'] = 30
params['earlystopping_patience'] = 50
params['max_epoch'] = 500
params['lam_T'] = 0.3
params['lam_Y'] = 0.1

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
splitting_list = [2931,3777]
dataset_list = ['VLE_zeotrope','VLE_azeotrope']
path_l = ['Tb_JCIM_normalized/all','Tc_JCIM_normalized/all']
name_l = ['Ensemble_0_Tb_JCIM_normalized_AGCNet','Ensemble_0_Tc_JCIM_normalized_AGCNet']

def main(params, net_params):
    model = AGCNetVLE6(net_params).to(device = 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr = params['init_lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'], 
                                                           patience=params['lr_schedule_patience'], verbose=False)
    t0 = time.time()
    per_epoch_time = []
    early_stopping = EarlyStopping(patience=params['earlystopping_patience'], path='checkpoint_ensemble' + params['Dataset'] + 'AGCNet_bayes_VLE_T' + '.pt')

    n_param = count_parameters(model)
    for epoch in range(params['max_epoch']):
        start = time.time()
        model, epoch_train_loss, epoch_train_metrics  = train_epoch_frag_VLE4(model, optimizer, scaling_y1,
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
                                                                            fetched_data.train_smiles_list_comp1,fetched_data.train_smiles_list_comp2,
                                                                            n_param,params['lam_Y'])

        epoch_val_loss, epoch_val_metrics = evaluate_frag_VLE4(model, scaling_y1, fetched_data.val_iter, 
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
                                                                                fetched_data.val_y1,
                                                                                fetched_data.val_smiles_list_comp1,fetched_data.val_smiles_list_comp2,
                                                                                n_param,params['lam_Y'])
        scheduler.step(epoch_val_loss)
        if optimizer.param_groups[0]['lr'] < params['min_lr']:
            print('\n! LR equal to min LR set.')
            break

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            break

    model = early_stopping.load_checkpoint(model)
    _, epoch_train_metrics = evaluate_frag_VLE4(model, scaling_y1,fetched_data.train_iter,
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
                                                fetched_data.train_smiles_list_comp1,fetched_data.train_smiles_list_comp2,
                                                n_param,params['lam_Y'])
    
    _, epoch_val_metrics = evaluate_frag_VLE4(model,scaling_y1,fetched_data.val_iter, 
                                                                                fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                                fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                                fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                                fetched_data.val_Tb_comp1,
                                                                                fetched_data.val_Tc_comp1,
                                                                                fetched_data.val_Tb_comp2,
                                                                                fetched_data.val_Tc_comp2,
                                                                                fetched_data.val_x1,
                                                                                fetched_data.val_y1,
                                                                                fetched_data.val_smiles_list_comp1,fetched_data.val_smiles_list_comp2,
                                                                                n_param,params['lam_Y'])
    _, epoch_test_metrics = evaluate_frag_VLE4(model,scaling_y1,fetched_data.test_iter, 
                                                                                fetched_data.test_batched_origin_graph_list_comp1, fetched_data.test_batched_origin_graph_list_comp2,
                                                                                fetched_data.test_batched_frag_graph_list_comp1, fetched_data.test_batched_frag_graph_list_comp2,
                                                                                fetched_data.test_batched_motif_graph_list_comp1, fetched_data.test_batched_motif_graph_list_comp2,
                                                                                fetched_data.test_Tb_comp1,
                                                                                fetched_data.test_Tc_comp1,
                                                                                fetched_data.test_Tb_comp2,
                                                                                fetched_data.test_Tc_comp2,
                                                                                fetched_data.test_x1,
                                                                                fetched_data.test_y1,
                                                                                fetched_data.test_smiles_list_comp1,fetched_data.test_smiles_list_comp2,n_param,params['lam_Y'])
    _, epoch_raw_metrics = evaluate_frag_VLE4(model,scaling_y1,fetched_data.all_iter, 
                                                                                fetched_data.all_batched_origin_graph_list_comp1, fetched_data.all_batched_origin_graph_list_comp2,
                                                                                fetched_data.all_batched_frag_graph_list_comp1, fetched_data.all_batched_frag_graph_list_comp2,
                                                                                fetched_data.all_batched_motif_graph_list_comp1, fetched_data.all_batched_motif_graph_list_comp2,
                                                                                fetched_data.all_Tb_comp1,
                                                                                fetched_data.all_Tc_comp1,
                                                                                fetched_data.all_Tb_comp2,
                                                                                fetched_data.all_Tc_comp2,
                                                                                fetched_data.all_x1,
                                                                                fetched_data.all_y1,fetched_data.all_smiles_list_comp1,fetched_data.all_smiles_list_comp2,n_param,params['lam_Y'])
    
    return epoch_train_metrics, epoch_val_metrics, epoch_test_metrics, epoch_raw_metrics

def Splitting_Main_MO(params, net_params):
    train_metrics, val_metrics, test_metrics, all_metrics = main(params, net_params)
    return -val_metrics.RMSE

def func_to_be_opt_MO(hidden_dim, depth, layers, decay, dropout, init_lr, lr_reduce_factor, lam_Y):
    net_params['hidden_dim'] = int(hidden_dim)
    net_params['layers'] = int(layers)
    net_params['depth'] = int(depth)
    params['weight_decay'] = 10 **(-int(decay))
    params['int_lr'] = 10 ** (-init_lr)
    params['lr_reduce_factor'] = 0.4 + 0.05 * int(lr_reduce_factor)
    params['lam_Y'] = lam_Y
    net_params['dropout'] = dropout
    return Splitting_Main_MO(params, net_params)

set_seed(seed = 1000)
torch.manual_seed(500)
for i in range(len(dataset_list)):
    params['Dataset'] = dataset_list[i]
    df, scaling_T, scaling_y1 = import_dataset(params)
    df['value'] = df['T']
    cache_file_path = os.path.realpath('./cache')
    if not os.path.exists(cache_file_path):
            os.mkdir(cache_file_path)
    cache_file = os.path.join(cache_file_path, params['Dataset'] + '_CCC')
    
    error_path = os.path.realpath('./error_log')
    if not os.path.exists(error_path):
        os.mkdir(error_path)
    error_log_path = os.path.join(error_path, '{}_{}'.format(params['Dataset'], time.strftime('%Y-%m-%d-%H-%M')) + '.csv')

    fragmentation = JT_SubGraph(scheme='MG_plus_reference')
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
    


    set_seed(seed =1000)
    rows = []
    file_path = os.path.realpath('./output')
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    save_file_path = os.path.join(file_path, '{}_{}_{}_{}_{}'.format('SOPT_UCB_5',params['Dataset'], 'AGCNetVLE', splitting_list[i], time.strftime('%Y-%m-%d-%H-%M')) + '.csv')

    split = VLE_Splitter2(allset)
    train_dataset, val_dataset, test_dataset, all_dataset = split.Random_Splitter(comp, seed = 100, frac_train = 0.8, frac_val = 0.9)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fraggraphs_VLE2, batch_size=len(train_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    val_loader = DataLoader(val_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(val_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    test_loader = DataLoader(test_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(test_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    raw_loader = DataLoader(all_dataset,collate_fn=collate_fraggraphs_VLE2, batch_size=len(all_dataset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))

    fetched_data = Azeotrope_PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=4)

    hpbounds = {'hidden_dim':(16,256.99),'depth':(1,5.99),'layers':(1,5.99),'decay':(0,6.99),'dropout':(0,0.5),'init_lr':(2,5),'lam_Y':(0.01,0.99),'lr_reduce_factor':(0,10.99)}
    mutating_optimizer = BayesianOptimization(f = func_to_be_opt_MO, pbounds = hpbounds, random_state = 250)
    mutating_optimizer.set_gp_params(alpha=1e-4, n_restarts_optimizer=1)
    utility = UtilityFunction(kind="ei", xi=0.01)
    mutating_optimizer.maximize(init_points = 5, n_iter = 50, acquisition_function = utility)
    lst = mutating_optimizer.space.keys
    lst.append('target')
    df = pd.DataFrame(columns = lst)
    for i, res in enumerate(mutating_optimizer.res):
        _dict = res['params']
        _dict['target'] = res['target']
        row = pd.DataFrame(_dict,index = [0])
        df = pd.concat([df,row],axis = 0,ignore_index = True, sort = False)
    df.to_csv(save_file_path, index = False)