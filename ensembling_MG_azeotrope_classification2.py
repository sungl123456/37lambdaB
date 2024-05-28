# 引入了沸点和临界温度预测，结果较未引入更优，说明了这两个性质的贡献是有必要的
from dataload.dataloading import import_dataset
from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.junctiontree_ring_encoder import frag
from utils.junctiontree_encoder import JT_SubGraph
from dataload.Azeotrope_dataset import Azeotrope_Dataset
from dataload.pair_dataset import pair_Dataset
from utils.mol2graph import smiles_2_bigraph
# from utils.splitter import Splitter
from utils.VLE_splitter import Azeotrope_Splitter, pair_Splitter2
from src.dgltools import collate_fraggraphs_VLE,collate_fraggraphs_pair
from utils.Earlystopping import EarlyStopping
from utils.count_parameters import count_parameters
from utils.VLE_piplines import Azeotrope_PreFetch,train_epoch_frag_azeotrope,predict_Tb_Tc, evaluate_frag_azeotrope,train_epoch_frag_clf,evaluate_frag_clf
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
from networks.AGC_pair import AGCNetpair
from networks.AGC_VLE import AGCNetVLE

from dataload.model_library import save_model, load_model, load_optimal_model

params = {}
net_params = {}
params['init_lr'] = 10 ** -2
params['min_lr'] = 1e-9
params['weight_decay'] = 10 ** -6
params['lr_reduce_factor'] = 0.75
params['lr_schedule_patience'] = 30
params['earlystopping_patience'] = 100
params['max_epoch'] = 200

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
splitting_seed = 2568
dataset_list = ['azeotrope_classification2']
path_l = ['Ensembles/Tb_VLE_normalized\MG_normalized','Ensembles/Tc_VLE_normalized\MG_normalized']
name_l = ['Ensemble_0_Tb_VLE_normalized_AGCNet','Ensemble_0_Tc_VLE_normalized_AGCNet']

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

comp = np.array(df)
all_size = len(comp)
print(all_size)
# # Contents of train_set, val_set will be disrupted
comp_train, comp_val, comp_test, comp_all = pair_Splitter2(comp, seed = splitting_seed, all_size = all_size)

trainset = pair_Dataset(df = comp_train, params = params, name = 'train',
                                smiles_2_graph = smiles_2_bigraph,
                                atom_featurizer = classic_atom_featurizer,
                                bond_featurizer = classic_bond_featurizer,
                                mol_featurizer = classic_mol_featurizer, 
                                cache_file_path = cache_file,
                                error_log=error_log_path,
                                load = False,
                                fragmentation=fragmentation)

valset = pair_Dataset(df = comp_val, params = params, name = 'val',
                                smiles_2_graph = smiles_2_bigraph,
                                atom_featurizer = classic_atom_featurizer,
                                bond_featurizer = classic_bond_featurizer,
                                mol_featurizer = classic_mol_featurizer, 
                                cache_file_path = cache_file,
                                error_log=error_log_path,
                                load = False,
                                fragmentation=fragmentation)

testset = pair_Dataset(df = comp_test, params = params, name = 'test',
                                smiles_2_graph = smiles_2_bigraph,
                                atom_featurizer = classic_atom_featurizer,
                                bond_featurizer = classic_bond_featurizer,
                                mol_featurizer = classic_mol_featurizer, 
                                cache_file_path = cache_file,
                                error_log=error_log_path,
                                load = False,
                                fragmentation=fragmentation)

allset = pair_Dataset(df = comp_all, params = params, name = 'all',
                                smiles_2_graph = smiles_2_bigraph,
                                atom_featurizer = classic_atom_featurizer,
                                bond_featurizer = classic_bond_featurizer,
                                mol_featurizer = classic_mol_featurizer, 
                                cache_file_path = cache_file,
                                error_log=error_log_path,
                                load = False,
                                fragmentation=fragmentation)

train_loader = DataLoader(trainset, collate_fn=collate_fraggraphs_VLE, batch_size=len(trainset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
val_loader = DataLoader(valset,collate_fn=collate_fraggraphs_VLE, batch_size=len(valset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
test_loader = DataLoader(testset,collate_fn=collate_fraggraphs_VLE, batch_size=len(testset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
raw_loader = DataLoader(allset,collate_fn=collate_fraggraphs_VLE, batch_size=len(allset), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))

fetched_data = Azeotrope_PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=3)
init_seed_list = [534]
torch.manual_seed(init_seed_list[0])
model = AGCNetVLE(net_params).to(device='cpu')
optimizer = torch.optim.Adam(model.parameters(), lr = params['init_lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                            patience=params['lr_schedule_patience'], verbose=False)

t0 = time.time()
per_epoch_time = []
early_stopping = EarlyStopping(patience=params['earlystopping_patience'], path='checkpoint_ensemble_' + params['Dataset'] + '_AGCNetVLE' + '_MG_ensemble' + '.pt')


with tqdm(range(params['max_epoch'])) as t:
    n_param = count_parameters(model)
    for epoch in t:
        t.set_description('Epoch %d' % epoch)
        start = time.time()
        model, epoch_train_loss, epoch_train_metrics  = train_epoch_frag_clf(model, optimizer, fetched_data.all_iter,
                                                                    fetched_data.all_batched_origin_graph_list_comp1, fetched_data.all_batched_origin_graph_list_comp2,
                                                                    fetched_data.all_batched_frag_graph_list_comp1, fetched_data.all_batched_frag_graph_list_comp2,
                                                                    fetched_data.all_batched_motif_graph_list_comp1, fetched_data.all_batched_motif_graph_list_comp2,
                                                                    fetched_data.all_Tb_comp1,
                                                                    fetched_data.all_Tc_comp1,
                                                                    fetched_data.all_Tb_comp2,
                                                                    fetched_data.all_Tc_comp2,
                                                                    fetched_data.all_targets_list)
        
        epoch_val_loss, epoch_val_metrics = evaluate_frag_clf(model,fetched_data.val_iter, 
                                                                    fetched_data.val_batched_origin_graph_list_comp1, fetched_data.val_batched_origin_graph_list_comp2,
                                                                    fetched_data.val_batched_frag_graph_list_comp1, fetched_data.val_batched_frag_graph_list_comp2,
                                                                    fetched_data.val_batched_motif_graph_list_comp1, fetched_data.val_batched_motif_graph_list_comp2,
                                                                    fetched_data.val_Tb_comp1,
                                                                    fetched_data.val_Tc_comp1,
                                                                    fetched_data.val_Tb_comp2,
                                                                    fetched_data.val_Tc_comp2,
                                                                    fetched_data.val_targets_list)
        
        t.set_postfix({'time': time.time() - start, 'lr': optimizer.param_groups[0]['lr'],
                                'train_score':  epoch_train_metrics.score,'val_score':epoch_val_metrics.score})
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

_, epoch_test_metrics = evaluate_frag_clf(model,fetched_data.test_iter, 
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

path = '/Ensembles/'+ dataset_list[0] + '/MG_ensemble_layer1'
name = '{}_{}_{}'.format('Ensemble_0', params['Dataset'], 'AGCNetVLE')
results = pd.Series({'init_seed': init_seed_list[0], 'seed': splitting_seed,
                    'train_score': epoch_train_metrics.score, 'val_score': epoch_val_metrics.score,
                    'test_score': epoch_test_metrics.score, 'all_score': epoch_raw_metrics.score,
                    'train_confusion_matirx': epoch_train_metrics.confusion,
                    'val_confusion_matrix': epoch_val_metrics.confusion,
                    'test_confusion_matrix': epoch_test_metrics.confusion,
                    'all_confusion_matrix': epoch_raw_metrics.confusion})

comments = ''
save_model(model, path, name, params, net_params, results, comments)
        