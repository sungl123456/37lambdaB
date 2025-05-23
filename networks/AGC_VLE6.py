# By Guanlun Sun(sungl123456@tju.edu.cn)


import dgl
import torch
import torch.nn as nn

from .AttentiveFP import Atom_AttentiveFP, Mol_AttentiveFP


class SingleHeadFragmentLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.AtomEmbedding = Atom_AttentiveFP(net_params)
        self.FragEmbedding = Mol_AttentiveFP(net_params)
        self.reset_parameters()

    def reset_parameters(self):
        self.AtomEmbedding.reset_parameters()
        self.FragEmbedding.reset_parameters()

    def forward(self, frag_graph, frag_node, frag_edge):
        # node_fragments: tensor: size(num_nodes_in_batch, num_features)
        node_fragments = self.AtomEmbedding(frag_graph, frag_node, frag_edge)
        super_frag, _ = self.FragEmbedding(frag_graph, node_fragments)
        return super_frag


class SingleHeadJunctionLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.project_motif = nn.Sequential(
            nn.Linear(net_params['hidden_dim'] + net_params['hidden_dim'], net_params['hidden_dim'], bias=True)
        )
        self.MotifEmbedding = Atom_AttentiveFP(net_params)
        self.GraphEmbedding = Mol_AttentiveFP(net_params)
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.project_motif:
            if isinstance(l, nn.Linear):
                l.reset_parameters()
        self.MotifEmbedding.reset_parameters()
        self.GraphEmbedding.reset_parameters()

    def forward(self, motif_graph, motif_node, motif_edge):
        motif_node = self.project_motif(motif_node)
        new_motif_node = self.MotifEmbedding(motif_graph, motif_node, motif_edge)
        super_new_graph, super_attention_weight = self.GraphEmbedding(motif_graph, new_motif_node)
        return super_new_graph, super_attention_weight

class AGCNetVLE6(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.dataset = net_params['Dataset']
        self.embedding_frag_node_lin_comp1 = nn.Sequential(
            nn.Linear(net_params['num_atom_type'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.embedding_frag_node_lin_comp2 = nn.Sequential(
            nn.Linear(net_params['num_atom_type'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.embedding_frag_edge_lin_comp1 = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.embedding_frag_edge_lin_comp2 = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.embedding_motif_node_lin_comp1 = nn.Sequential(
            nn.Linear(net_params['frag_dim'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.embedding_motif_node_lin_comp2 = nn.Sequential(
            nn.Linear(net_params['frag_dim'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.embedding_motif_edge_lin_comp1 = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.embedding_motif_edge_lin_comp2 = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )

        self.num_heads = net_params['num_heads']
        self.fragment_heads_comp1 = nn.ModuleList([SingleHeadFragmentLayer(net_params) for _ in range(self.num_heads)])
        self.junction_heads_comp1 = nn.ModuleList([SingleHeadJunctionLayer(net_params) for _ in range(self.num_heads)])
        self.fragment_heads_comp2 = nn.ModuleList([SingleHeadFragmentLayer(net_params) for _ in range(self.num_heads)])
        self.junction_heads_comp2 = nn.ModuleList([SingleHeadJunctionLayer(net_params) for _ in range(self.num_heads)])

        self.frag_attend_comp1 = nn.Sequential(
            nn.Linear(self.num_heads * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.frag_attend_comp2 = nn.Sequential(
            nn.Linear(self.num_heads * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.motif_attend_comp1 = nn.Sequential(
            nn.Linear(self.num_heads * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.motif_attend_comp2 = nn.Sequential(
            nn.Linear(self.num_heads * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.linear_predict = nn.Sequential(
            nn.Dropout(net_params['dropout']),
            nn.Linear(2 * net_params['hidden_dim'] + 5, net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
        )
        self.mean_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(net_params['hidden_dim'], 1, bias=True)
        )
        self.var_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(net_params['hidden_dim'], 1, bias=True))
        self.reset_parameters()
    
    def reset_parameters(self):
        for fragment_layer in self.fragment_heads_comp1:
            fragment_layer.reset_parameters()
        for fragment_layer in self.fragment_heads_comp2:
            fragment_layer.reset_parameters()
        for junction_layer in self.junction_heads_comp1:
            junction_layer.reset_parameters()
        for junction_layer in self.junction_heads_comp2:
            junction_layer.reset_parameters()
        for layer in self.linear_predict:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.mean_head:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.var_head:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
    
    def forward(self, origin_graph_comp1, origin_graph_comp2, 
                origin_node_comp1, origin_node_comp2, 
                origin_edge_comp1, origin_edge_comp2, 
                frag_graph_comp1, frag_graph_comp2, 
                frag_node_comp1, frag_node_comp2, 
                frag_edge_comp1, frag_edge_comp2, 
                motif_graph_comp1, motif_graph_comp2, 
                motif_node_comp1, motif_node_comp2, 
                motif_edge_comp1, motif_edge_comp2,
                Tb_comp1, Tc_comp1, Tb_comp2, Tc_comp2, x1,
                get_descriptors=False, get_attention=False):
        # Fragments Layer:
        frag_node_comp1 = frag_node_comp1.float()
        frag_edge_comp1 = frag_edge_comp1.float()
        frag_node_comp1 = self.embedding_frag_node_lin_comp1(frag_node_comp1)
        frag_edge_comp1 = self.embedding_frag_edge_lin_comp1(frag_edge_comp1)
        frag_heads_out_comp1 = [frag_block(frag_graph_comp1, frag_node_comp1, frag_edge_comp1) for frag_block in self.fragment_heads_comp1]
        graph_motif_comp1 = self.frag_attend_comp1(torch.cat(frag_heads_out_comp1, dim=-1))
        motif_graph_comp1.ndata['feats'] = graph_motif_comp1
        # Junction Tree Layer:
        motif_edge_comp1 = motif_edge_comp1.float()
        motif_node_comp1 = self.embedding_motif_node_lin_comp1(motif_node_comp1)
        motif_edge_comp1 = self.embedding_motif_edge_lin_comp1(motif_edge_comp1)
        motif_node_comp1 = torch.cat([graph_motif_comp1, motif_node_comp1], dim=-1)
        junction_graph_heads_out_comp1 = []
        junction_attention_heads_out_comp1 = []
        for single_head in self.junction_heads_comp1:
            single_head_new_graph_comp1, single_head_attention_weight_comp1 = single_head(motif_graph_comp1, motif_node_comp1, motif_edge_comp1)
            junction_graph_heads_out_comp1.append(single_head_new_graph_comp1)
            junction_attention_heads_out_comp1.append(single_head_attention_weight_comp1)
        # Fragments Layer:
        frag_node_comp2 = frag_node_comp2.float()
        frag_edge_comp2 = frag_edge_comp2.float()
        frag_node_comp2 = self.embedding_frag_node_lin_comp2(frag_node_comp2)
        frag_edge_comp2 = self.embedding_frag_edge_lin_comp2(frag_edge_comp2)
        frag_heads_out_comp2 = [frag_block(frag_graph_comp2, frag_node_comp2, frag_edge_comp2) for frag_block in self.fragment_heads_comp2]
        graph_motif_comp2 = self.frag_attend_comp2(torch.cat(frag_heads_out_comp2, dim=-1))
        motif_graph_comp2.ndata['feats'] = graph_motif_comp2
        # Junction Tree Layer:
        motif_edge_comp2 = motif_edge_comp2.float()
        motif_node_comp2 = self.embedding_motif_node_lin_comp2(motif_node_comp2)
        motif_edge_comp2 = self.embedding_motif_edge_lin_comp2(motif_edge_comp2)
        motif_node_comp2 = torch.cat([graph_motif_comp2, motif_node_comp2], dim=-1)
        junction_graph_heads_out_comp2 = []
        junction_attention_heads_out_comp2 = []
        for single_head in self.junction_heads_comp2:
            single_head_new_graph_comp2, single_head_attention_weight_comp2 = single_head(motif_graph_comp2, motif_node_comp2, motif_edge_comp2)
            junction_graph_heads_out_comp2.append(single_head_new_graph_comp2)
            junction_attention_heads_out_comp2.append(single_head_attention_weight_comp2)

        super_new_graph_comp1 = torch.relu(torch.mean(torch.stack(junction_graph_heads_out_comp1,dim = 1),dim=1))
        super_new_graph_comp2 = torch.relu(torch.mean(torch.stack(junction_graph_heads_out_comp2,dim = 1),dim=1))
        super_attention_weight_comp1 = torch.relu(torch.mean(torch.stack(junction_attention_heads_out_comp1,dim = 1),dim=1))
        super_attention_weight_comp2 = torch.relu(torch.mean(torch.stack(junction_attention_heads_out_comp2,dim = 1),dim=1))
        
        concat_features = torch.cat([super_new_graph_comp1, super_new_graph_comp2,Tb_comp1,Tc_comp1,Tb_comp2,Tc_comp2,x1],dim=-1)
        descriptors = self.linear_predict(concat_features)
        output= torch.sigmoid(self.mean_head(descriptors))
        log_var = self.var_head(descriptors)
        if get_attention:
            motif_graph_comp1.ndata['attention_weight'] = super_attention_weight_comp1
            motif_graph_comp2.ndata['attention_weight'] = super_attention_weight_comp2
            attention_list_array_comp1 = []
            attention_list_array_comp2 = []
            for g in dgl.unbatch(motif_graph_comp1):
                attention_list_array_comp1.append(g.ndata['attention_weight'].detach().to(device='cpu').numpy())
            for g in dgl.unbatch(motif_graph_comp2):
                attention_list_array_comp2.append(g.ndata['attention_weight'].detach().to(device='cpu').numpy())
            return output, log_var, attention_list_array_comp1, attention_list_array_comp2
        if get_descriptors:
            return output, log_var, super_new_graph_comp1, super_new_graph_comp2
        else:
            return output, log_var
    
    def loss(self, scores, targets, log_var, lam, epsilon=1e-6):
        mse_loss = (scores - targets) ** 2
        sigma_squared = torch.exp(log_var) + epsilon
        loss = lam * mse_loss + (1-lam) * (scores-targets) **2 / sigma_squared
        return loss.mean()

        
