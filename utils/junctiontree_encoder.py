# -*- coding: utf-8 -*-
# @Time    : 2022/4/3 11:13
# @Author  : FAN FAN
# @Site    : 
# @File    : junctiontree_encoder.py
# @Software: PyCharm
import numpy as np
from rdkit import Chem
import os
import pandas as pd
import dgl
import torch


class JT_SubGraph():
    def __init__(self, scheme):
        #self.smiles_list = smiles_list
        path = os.path.join('.//dataset', scheme + '.csv')
        data_from = os.path.realpath(path)
        df = pd.read_csv(data_from)
        pattern = np.array([df['First-Order Group'], df['SMARTs'], df['Priority']])
        self.sorted_pattern = pattern[:, np.argsort(pattern[2, :])]
        self.frag_name_list = list(dict.fromkeys(self.sorted_pattern[0, :]))
        self.frag_dim = len(self.frag_name_list)
        #self.max_mol_size = np.max([Chem.MolFromSmiles(smiles).GetNumAtoms() for _, smiles in enumerate(self.smiles_list)])

    def fragmentation(self, graph, mol):
        """Fragmentation
        Parameters
        ----------
        graph : DGLGraph
            DGLGraph for a batch of graphs.
        index : int
            Index of molecule in dataset.
        """
        pat_list = []
        mol_size = mol.GetNumAtoms()
        for patt in self.sorted_pattern[1, :]:
            pat = Chem.MolFromSmarts(patt)#把基团转为mol
            pat_list.append(list(mol.GetSubstructMatches(pat)))#跟基团有重复结构的原子位置输出
            # pat_list: list of lists containing atom index tuples, e.g. [[(2,), (3,), (5,), (6,)], [(0,1)], [(4, 7]]

        num_atoms = mol.GetNumAtoms()
        atom_idx_list = [i for i in range(num_atoms)]
        # hit_ats: dictionary of each fragments and their atom idx, values in np.array
        hit_ats = {}
        frag_flag = []
        prior_set = set()
        k = 0

        for idx, key in enumerate(self.sorted_pattern[0, :]):#基团名称
            frags = pat_list[idx]
            if frags:
                # print(frags)
                # check overlapping on same subgroups: e.x. trimethylamine 2*CH3, 1*CH3N
                for i, item in enumerate(frags):
                    item_set = set(item)
                    new_frags = frags[:i] + frags[i + 1:]
                    left_set = set(sum(new_frags, ()))
                    if not item_set.isdisjoint(left_set):
                        frags = new_frags
                for _, frag in enumerate(frags):
                    # cur_idx = set(list(itertools.chain(*pat_list[idx])))
                    frag_set = set(frag)  # tuple -> set
                    if prior_set.isdisjoint(frag_set):#判断prior_set和目前的frag_set中的原子位置是否有重合，有重合返回False
                        ats = frag_set
                    else:
                        ats = {}
                    if ats:#不是空的集合则执行以下程序，我猜这步可能是为了避免已经被之前的基团包进去的原子位置不被新的基团识别.
                        adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(mol)[np.newaxis, :, :]#一个多了一个维度的分子的邻接矩阵
                        if k == 0:
                            adj_mask = adjacency_origin#
                            atom_mask = np.zeros((1, mol_size))#一个全是0的数组，尺寸和分子的原子数目相同
                            frag_features = np.asarray(list(map(lambda s: float(key == s), self.frag_name_list))) #产生一个新的数组，map()对应位置产生映射，把基团对应到219个里面
                        else:
                            adj_mask = np.vstack((adj_mask, adjacency_origin))# np.vstack沿着行方向叠一个新的数组，需要与原先的数组纵向维度相同
                            atom_mask = np.vstack((atom_mask, np.zeros((1, mol_size))))
                            frag_features = np.vstack((frag_features,np.asarray(list(map(lambda s: float(key == s), self.frag_name_list)))))
                        if key not in hit_ats.keys():
                            hit_ats[key] = np.asarray(list(ats))
                        else:
                            hit_ats[key] = np.vstack((hit_ats[key], np.asarray(list(ats))))
                        ignores = list(set(atom_idx_list) - set(ats))

                        adj_mask[k, ignores, :] = 0
                        adj_mask[k, :, ignores] = 0#相当于将所有的基团的邻接矩阵的第ats行和第ats列变成0
                        atom_mask[k, list(ats)] = 1
                        frag_flag.append(key)#把mol包含的基团名字存进去e.g.['ACC=O', 'ACH', 'ACH', 'ACH', 'ACH', 'ACH', 'CH3']
                        k += 1
                        prior_set.update(ats)#逐步将上述基团包含所有的原子加进来 e.g. {0,1,2,3...}

        # unknown fragments:
        unknown_ats = list(set(atom_idx_list) - prior_set)#如果有原子没被现有的一次基团包进去，那么用unknown表示
        if len(unknown_ats) > 0:
            for i, at in enumerate(unknown_ats):
                if k == 0:#这之前的k没有归0过，所以应该是会加到上面一次基团识别得到的矩阵中去
                    if num_atoms == 1:
                        adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(mol)[np.newaxis, :, :]
                    adj_mask = adjacency_origin
                    atom_mask = np.zeros((1, mol_size))
                else:
                    # adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(m)[np.newaxis, :, :]
                    adj_mask = np.vstack((adj_mask, adjacency_origin))
                    atom_mask = np.vstack((atom_mask, np.zeros((1, mol_size))))
                if 'unknown' not in hit_ats.keys():
                    hit_ats['unknown'] = np.asarray(at)
                else:
                    hit_ats['unknown'] = np.vstack((hit_ats['unknown'], np.asarray(at)))
                ignores = list(set(atom_idx_list) - set([at]))
                # print(prior_idx)
                if num_atoms != 1:
                    adj_mask[k, ignores, :] = 0
                    adj_mask[k, :, ignores] = 0
                atom_mask[k, at] = 1
                frag_flag.append('unknown')
                if num_atoms != 1:
                    frag_features = np.vstack(
                        (frag_features, np.asarray(list(map(lambda s: float('unknown' == s), self.frag_name_list)))))
                else:
                    frag_features = np.asarray(list(map(lambda s: float('unknown' == s), self.frag_name_list)))
                k += 1
        # adj_mask: size: [num of fragments, max num of atoms, max num of atoms]
        # atom_mask: size: [num of fragments, max num of atoms]
        # adjacency_fragments: size: [max num of atoms, max num of atoms]

        adjacency_fragments = adj_mask.sum(axis=0)#把减出来的矩阵中的非零数的位置按照行索引和列索引分别输出成两个array
        idx1, idx2 = (adjacency_origin.squeeze(0) - adjacency_fragments).nonzero()#将两个array转为列表再变为一个列表中的两个元素

        # idx_tuples: list of tuples, idx of begin&end atoms on each new edge
        idx_tuples = list(zip(idx1.tolist(), idx2.tolist()))
        # remove reverse edges
        # idx_tuples = list(set([tuple(sorted(item)) for item in idx_tuples]))
        rm_edge_ids_list = []
        for i, item in enumerate(idx_tuples):
            try:
                rm_edge_ids = graph.edge_ids(item[0], item[1])
            except:
                #rm_edge_ids = graph.edge_ids(item[1], item[0])
                continue
            rm_edge_ids_list.append(rm_edge_ids)

        frag_graph = dgl.remove_edges(graph, rm_edge_ids_list)

        num_motifs = atom_mask.shape[0]
        motif_graph = dgl.DGLGraph()
        motif_graph.add_nodes(num_motifs)

        adjacency_motifs, idx_tuples, motif_graph = self.build_adjacency_motifs(atom_mask, idx_tuples, motif_graph)#往motif_graph里添加边向量
        #idx_tuples = list(set([tuple(sorted(item)) for item in idx_tuples]))

        if frag_features.ndim == 1:
            frag_features = frag_features.reshape(-1, 1).transpose()#跟前面那个一样，没变化

        motif_graph.ndata['feat'] = torch.Tensor(frag_features)
        motif_graph.ndata['atom_mask'] = torch.Tensor(atom_mask)

        edge_features = graph.edata['feat']
        add_edge_feats_ids_list = []
        for i, item in enumerate(idx_tuples):
            try:
                add_edge_feats_ids = graph.edge_ids(item[0], item[1])
            except:
                #add_edge_feats_ids = graph.edge_ids(item[1], item[0])
                continue
            add_edge_feats_ids_list.append(add_edge_feats_ids)
        if num_atoms != 1:
            motif_edge_features = edge_features[add_edge_feats_ids_list, :]
            try:
                motif_graph.edata['feat'] = motif_edge_features

                frag_graph_list = self.rebuild_frag_graph(frag_graph, motif_graph)
                motif_graph.ndata.pop('atom_mask')
                return frag_graph_list, motif_graph, atom_mask, frag_flag
            except:
                frag_graph_list = self.rebuild_frag_graph(frag_graph, motif_graph)
                return frag_graph_list, None
        else:
            frag_graph_list = self.rebuild_frag_graph(frag_graph, motif_graph)
            motif_graph.ndata.pop('atom_mask')
            return frag_graph_list, motif_graph, atom_mask, frag_flag

    def atom_locate_frag(self, atom_mask, atom):
        return atom_mask[:, atom].tolist().index(1)

    def frag_locate_atom(self, atom_mask, frag):
        return atom_mask[frag, :].nonzero()[0].tolist()

    def build_adjacency_motifs(self, atom_mask, idx_tuples, motif_graph):
        k = atom_mask.shape[0]
        duplicate_bond = []
        adjacency_motifs = np.zeros((k, k)).astype(int)
        motif_edge_begin = list(map(lambda x: self.atom_locate_frag(atom_mask, x[0]), idx_tuples))
        motif_edge_end = list(map(lambda x: self.atom_locate_frag(atom_mask, x[1]), idx_tuples))
        #adjacency_motifs[new_edge_begin, new_edge_end] = 1
        # eliminate duplicate bond in triangle substructure
        for idx1, idx2 in zip(motif_edge_begin, motif_edge_end):
            if adjacency_motifs[idx1, idx2] == 0:
                adjacency_motifs[idx1, idx2] = 1
                motif_graph.add_edges(idx1, idx2)
            else:
                rm_1 = self.frag_locate_atom(atom_mask, idx1)
                rm_2 = self.frag_locate_atom(atom_mask, idx2)
                if isinstance(rm_1, int):
                    rm_1 = [rm_1]
                if isinstance(rm_2, int):
                    rm_2 = [rm_2]
                for i in rm_1:
                    for j in rm_2:
                        duplicate_bond.extend([tup for tup in idx_tuples if tup == (i, j)])
        if duplicate_bond:
            idx_tuples.remove(duplicate_bond[0])
            idx_tuples.remove(duplicate_bond[2])
        return adjacency_motifs, idx_tuples, motif_graph

    def rebuild_frag_graph(self, frag_graph, motif_graph):
        num_motifs = motif_graph.num_nodes()
        frag_graph_list = []
        for idx_motif in range(num_motifs):
            #new_frag_graph = dgl.DGLGraph()
            coord = motif_graph.nodes[idx_motif].data['atom_mask'].nonzero()
            idx_list = []
            for idx_node in coord:
                idx_list.append(idx_node[1])
            new_frag_graph = dgl.node_subgraph(frag_graph, idx_list)
            frag_graph_list.append(new_frag_graph)
        return frag_graph_list
