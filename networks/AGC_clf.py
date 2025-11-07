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

class AGCNetCLF(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.dataset = net_params['Dataset']
        self.embedding_frag_node_lin_comp1 = nn.Sequential(
            nn.Linear(net_params['num_atom_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.embedding_frag_node_lin_comp2 = nn.Sequential(
            nn.Linear(net_params['num_atom_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.embedding_frag_edge_lin_comp1 = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.embedding_frag_edge_lin_comp2 = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.embedding_motif_node_lin_comp1 = nn.Sequential(
            nn.Linear(net_params['frag_dim'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.embedding_motif_node_lin_comp2 = nn.Sequential(
            nn.Linear(net_params['frag_dim'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.embedding_motif_edge_lin_comp1 = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.embedding_motif_edge_lin_comp2 = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
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
        self.linear_predict1_1 = nn.Sequential(
            nn.Linear(net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.ReLU()
        )
        self.linear_predict1_2 = nn.Sequential(
            nn.Linear(net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.ReLU()
        )
        self.linear_predict1_3 = nn.Sequential(
            nn.Linear(4, net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.ReLU()
        )
        self.linear_predict3 = nn.Sequential(
            nn.Linear(3*net_params['hidden_dim'], net_params['hidden_dim'], bias=True)
        )  
        self.linear_predict4 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(net_params['hidden_dim'], 1, bias=True)
        )
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
        for layer in self.linear_predict1_1:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.linear_predict1_2:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.linear_predict1_3:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()    
        for layer in self.linear_predict3:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.linear_predict4:
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
                Tb_comp1,Tc_comp1, Tb_comp2, Tc_comp2,
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
        
        concat_Tb_Tc = torch.cat([Tb_comp1,Tc_comp1,Tb_comp2,Tc_comp2],dim = -1)
        descriptors1_1 = self.linear_predict1_1(super_new_graph_comp1)
        descriptors1_2 = self.linear_predict1_2(super_new_graph_comp2)
        descriptors1_3 = self.linear_predict1_3(concat_Tb_Tc) 
        concat_features = torch.cat([descriptors1_1,descriptors1_2,descriptors1_3],dim=-1)
        descriptors3 = self.linear_predict3(concat_features)
        output = torch.sigmoid(self.linear_predict4(descriptors3))
        if get_attention:
            motif_graph_comp1.ndata['attention_weight'] = super_attention_weight_comp1
            motif_graph_comp2.ndata['attention_weight'] = super_attention_weight_comp2
            attention_list_array_comp1 = []
            attention_list_array_comp2 = []
            for g in dgl.unbatch(motif_graph_comp1):
                attention_list_array_comp1.append(g.ndata['attention_weight'].detach().to(device='cpu').numpy())
            for g in dgl.unbatch(motif_graph_comp2):
                attention_list_array_comp2.append(g.ndata['attention_weight'].detach().to(device='cpu').numpy())
            return output, attention_list_array_comp1, attention_list_array_comp2
        if get_descriptors:
            return output, super_new_graph_comp1, super_new_graph_comp2
        else:
            return output
    
    def loss(self, scores, targets):
        loss = nn.BCELoss()(scores,targets)
        return loss

        
