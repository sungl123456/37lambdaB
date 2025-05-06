# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 13:17
# @Author  : FAN FAN
# @Site    : 
# @File    : dgltools.py
# @Software: PyCharm
import numpy as np
import torch
import dgl


def collate_molgraphs(samples):
    origin_graphs, targets, smiles = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)
    batched_origin_graph = dgl.batch(origin_graphs)
    return batched_origin_graph, targets, smiles


def collate_fraggraphs(samples):
    # origin_graphs, motif_graphs: list of graphs:
    origin_graphs, frag_graphs, motif_graphs, _, targets, smiles = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)

    batched_origin_graph = dgl.batch(origin_graphs)
    batched_frag_graph = dgl.batch(frag_graphs)
    batched_motif_graph = dgl.batch(motif_graphs)
    return batched_origin_graph, batched_frag_graph, batched_motif_graph, targets, smiles

def collate_fraggraphs_pair(samples):
    origin_graphs_comp1, origin_graphs_comp2, frag_graphs_comp1, frag_graphs_comp2, motif_graphs_comp1, motif_graphs_comp2, _, _, targets, smiles1, smiles2 = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)
    batched_origin_graph_comp1 = dgl.batch(origin_graphs_comp1)
    batched_origin_graph_comp2 = dgl.batch(origin_graphs_comp2)
    batched_frag_graph_comp1 = dgl.batch(frag_graphs_comp1)
    batched_frag_graph_comp2 = dgl.batch(frag_graphs_comp2)
    batched_motif_graph_comp1 = dgl.batch(motif_graphs_comp1)
    batched_motif_graph_comp2 = dgl.batch(motif_graphs_comp2)
    return batched_origin_graph_comp1, batched_origin_graph_comp2, batched_frag_graph_comp1, batched_frag_graph_comp2, batched_motif_graph_comp1, batched_motif_graph_comp2, targets, smiles1, smiles2

def collate_fraggraphs_VLE(samples):
    origin_graphs_comp1, origin_graphs_comp2, frag_graphs_comp1, frag_graphs_comp2, motif_graphs_comp1, motif_graphs_comp2, _, _, targets, smiles1, smiles2, Tb_comp1, Tc_comp1, Tb_comp2, Tc_comp2 = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)
    batched_origin_graph_comp1 = dgl.batch(origin_graphs_comp1)
    batched_origin_graph_comp2 = dgl.batch(origin_graphs_comp2)
    batched_frag_graph_comp1 = dgl.batch(frag_graphs_comp1)
    batched_frag_graph_comp2 = dgl.batch(frag_graphs_comp2)
    batched_motif_graph_comp1 = dgl.batch(motif_graphs_comp1)
    batched_motif_graph_comp2 = dgl.batch(motif_graphs_comp2)
    Tb_comp1 = torch.tensor(np.array(Tb_comp1)).unsqueeze(1)
    Tc_comp1 = torch.tensor(np.array(Tc_comp1)).unsqueeze(1)
    Tb_comp2 = torch.tensor(np.array(Tb_comp2)).unsqueeze(1)
    Tc_comp2 = torch.tensor(np.array(Tc_comp2)).unsqueeze(1)
    return batched_origin_graph_comp1, batched_origin_graph_comp2, batched_frag_graph_comp1, batched_frag_graph_comp2, batched_motif_graph_comp1, batched_motif_graph_comp2, targets, smiles1, smiles2, Tb_comp1, Tc_comp1, Tb_comp2, Tc_comp2

def collate_fraggraphs_VLE2(samples):
    origin_graphs_comp1, origin_graphs_comp2, frag_graphs_comp1, frag_graphs_comp2, motif_graphs_comp1, motif_graphs_comp2, _, _, smiles1, smiles2, Tb_comp1, Tc_comp1, Tb_comp2, Tc_comp2, x1, T_paodian, y1 = map(list, zip(*samples))
    batched_origin_graph_comp1 = dgl.batch(origin_graphs_comp1)
    batched_origin_graph_comp2 = dgl.batch(origin_graphs_comp2)
    batched_frag_graph_comp1 = dgl.batch(frag_graphs_comp1)
    batched_frag_graph_comp2 = dgl.batch(frag_graphs_comp2)
    batched_motif_graph_comp1 = dgl.batch(motif_graphs_comp1)
    batched_motif_graph_comp2 = dgl.batch(motif_graphs_comp2)
    Tb_comp1 = torch.tensor(np.array(Tb_comp1)).unsqueeze(1)
    Tc_comp1 = torch.tensor(np.array(Tc_comp1)).unsqueeze(1)
    Tb_comp2 = torch.tensor(np.array(Tb_comp2)).unsqueeze(1)
    Tc_comp2 = torch.tensor(np.array(Tc_comp2)).unsqueeze(1)
    batched_x1 = torch.tensor(np.array(x1)).unsqueeze(1)
    batched_T = torch.tensor(np.array(T_paodian)).unsqueeze(1)
    batched_y1 = torch.tensor(np.array(y1)).unsqueeze(1)
    return batched_origin_graph_comp1, batched_origin_graph_comp2, batched_frag_graph_comp1, batched_frag_graph_comp2, batched_motif_graph_comp1, batched_motif_graph_comp2, smiles1, smiles2, Tb_comp1, Tc_comp1, Tb_comp2, Tc_comp2, batched_x1, batched_T, batched_y1


def collate_gcgatgraphs(samples):
    # origin_graphs, motif_graphs: list of graphs:
    origin_graphs, frag_graphs, motif_graphs, channel_graphs, targets, smiles = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)

    batched_origin_graph = dgl.batch(origin_graphs)
    batched_frag_graph = dgl.batch(frag_graphs)
    batched_motif_graph = dgl.batch(motif_graphs)
    batched_channel_graph = dgl.batch(channel_graphs)

    batched_index_list = []
    batch_len = batched_channel_graph.batch_size
    for i in range(batch_len):
        batched_index_list.append(i)
        batched_index_list.append(i + batch_len)
        batched_index_list.append(i + 2 * batch_len)

    return batched_origin_graph, batched_frag_graph, batched_motif_graph, batched_channel_graph, batched_index_list, targets, smiles


def collate_fraggraphs_backup(samples):
    # origin_graphs, motif_graphs: list of graphs:
    origin_graphs, frag_graphs, motif_graphs, targets, smiles = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)
    frag_graphs_list = []
    for item in frag_graphs:
        frag_graphs_list.extend(item)

    batched_origin_graph = dgl.batch(origin_graphs)
    batched_frag_graph = dgl.batch(frag_graphs_list)
    batched_motif_graph = dgl.batch(motif_graphs)
    return batched_origin_graph, batched_frag_graph, batched_motif_graph, targets, smiles

