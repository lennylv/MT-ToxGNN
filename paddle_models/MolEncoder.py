import numpy as np

import paddle
import paddle.nn as nn
import pgl
from pgl.nn import GraphPool

from paddle_models.GNN_blocks import GIN, MeanPool, GraphNorm
from paddle_models.compound_encoder import AtomEmbedding, BondEmbedding, BondFloatRBF, VanBondFloatRBF, BondAngleFloatRBF

class GNNBlock(nn.Layer):

    def __init__(self, embed_dim, dropout_rate, last_act):
        super(GNNBlock, self).__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act

        self.gnn = GIN(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.graph_norm = GraphNorm()
        if last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, graph, node_hidden, edge_hidden):
        """tbd"""
        out = self.gnn(graph, node_hidden, edge_hidden)
        out = self.norm(out)
        out = self.graph_norm(graph, out)
        if self.last_act:
            out = self.act(out)
        out = self.dropout(out)
        out = out + node_hidden
        return out

class vdWGraph(nn.Layer):
    def __init__(self, model_config={}):
        super(vdWGraph, self).__init__()

        self.embed_dim = model_config.get('embed_dim', 128)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        self.layer_num = model_config.get('layer_num', 8)
        self.readout = model_config.get('readout', 'mean')
        self.layer_select = model_config.get('layer_select', -1)

        self.atom_names = model_config['atom_names']
        self.bond_names = model_config['bond_names']
        self.bond_float_names = model_config['bond_float_names']
        self.edge_van_names = model_config['atom_van_bond_names']

        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim)
        self.init_bond_van_float_rbf = VanBondFloatRBF(self.edge_van_names, self.embed_dim)
        
        self.bond_embedding_list = nn.LayerList()
        self.bond_float_rbf_list = nn.LayerList()

        self.atom_van_bond_rbf_list = nn.LayerList()

        self.atom_bond_block_list = nn.LayerList()
        self.atom_van_bond_list = nn.LayerList()

        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(
                    BondEmbedding(self.bond_names, self.embed_dim))
            self.bond_float_rbf_list.append(
                    BondFloatRBF(self.bond_float_names, self.embed_dim))

            self.atom_van_bond_rbf_list.append(
                    VanBondFloatRBF(self.edge_van_names, self.embed_dim))
            
            self.atom_bond_block_list.append(
                    GNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
            self.atom_van_bond_list.append(
                    GNNBlock(self.embed_dim, self.dropout_rate, last_act=layer_id != self.layer_num - 1))

        if self.readout == 'mean':
            self.graph_pool = MeanPool()
        else:
            self.graph_pool = pgl.nn.GraphPool(pool_type=self.readout)

        print('[lulu] embed_dim:%s' % self.embed_dim)
        print('[lulu] dropout_rate:%s' % self.dropout_rate)
        print('[lulu] layer_num:%s' % self.layer_num)
        print('[lulu] readout:%s' % self.readout)
        print('[lulu] atom_names:%s' % str(self.atom_names))
        print('[lulu] bond_names:%s' % str(self.bond_names))
        print('[lulu] bond_float_names:%s' % str(self.bond_float_names))

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(self, atom_bond_graph, van_graph):
        """
        Build the network.
        """
        node_hidden = self.init_atom_embedding(atom_bond_graph.node_feat)
        bond_embed = self.init_bond_embedding(atom_bond_graph.edge_feat)
        edge_hidden = bond_embed + self.init_bond_float_rbf(atom_bond_graph.edge_feat)

        van_edge_hidden = self.init_bond_van_float_rbf( van_graph.edge_feat )

        node_hidden_list = [node_hidden]
        van_node_hidden_list = [node_hidden]

        edge_hidden_list = [edge_hidden]
        van_edge_hidden_list = [van_edge_hidden]

        for layer_id in range(self.layer_num):
            node_hidden = self.atom_bond_block_list[layer_id](
                    atom_bond_graph,
                    node_hidden_list[layer_id],
                    edge_hidden_list[layer_id])
            
            
            van_node_hidden = self.atom_van_bond_list[layer_id](
                    van_graph,
                    van_node_hidden_list[layer_id],
                    van_edge_hidden_list[layer_id])
            
            cur_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph.edge_feat)
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](atom_bond_graph.edge_feat)

            cur_van_edge_hidden = self.atom_van_bond_rbf_list[layer_id](van_graph.edge_feat)
            
            node_hidden_list.append(node_hidden)
            van_node_hidden_list.append(van_node_hidden)

            edge_hidden_list.append(cur_edge_hidden)
            van_edge_hidden_list.append(cur_van_edge_hidden)
        
        node_repr = node_hidden_list[self.layer_select]
        van_node_repr = van_node_hidden_list[self.layer_select]

        edge_repr = edge_hidden_list[self.layer_select]
        van_edge_repr = van_edge_hidden_list[self.layer_select]

        graph_repr = self.graph_pool(atom_bond_graph, node_repr)
        van_graph_repr = self.graph_pool(van_graph, van_node_repr)

        node_repr = node_repr + van_node_repr
        graph_repr = graph_repr + van_graph_repr

        return node_repr, edge_repr, graph_repr

class GNE(nn.Layer):
    def __init__(self, model_config={}):
        super(GNE, self).__init__()

        self.embed_dim = model_config.get('embed_dim', 128)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        self.layer_num = model_config.get('layer_num', 8)
        self.readout = model_config.get('readout', 'mean')

        self.atom_names = model_config['atom_names']
        self.bond_names = model_config['bond_names']
        self.bond_float_names = model_config['bond_float_names']
        self.edge_van_names = model_config['atom_van_bond_names']

        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim)
        self.init_bond_van_float_rbf = VanBondFloatRBF(self.edge_van_names, self.embed_dim)
        
        self.bond_embedding_list = nn.LayerList()
        self.bond_float_rbf_list = nn.LayerList()

        self.atom_van_bond_rbf_list = nn.LayerList()

        self.atom_bond_block_list = nn.LayerList()
        self.atom_van_bond_list = nn.LayerList()

        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(
                    BondEmbedding(self.bond_names, self.embed_dim))
            self.bond_float_rbf_list.append(
                    BondFloatRBF(self.bond_float_names, self.embed_dim))

            self.atom_van_bond_rbf_list.append(
                    VanBondFloatRBF(self.edge_van_names, self.embed_dim))
            
            self.atom_bond_block_list.append(
                    GNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
            self.atom_van_bond_list.append(
                    GNNBlock(self.embed_dim, self.dropout_rate, last_act=layer_id != self.layer_num - 1))

        if self.readout == 'mean':
            self.graph_pool = MeanPool()
        else:
            self.graph_pool = pgl.nn.GraphPool(pool_type=self.readout)

        print('[lulu] embed_dim:%s' % self.embed_dim)
        print('[lulu] dropout_rate:%s' % self.dropout_rate)
        print('[lulu] layer_num:%s' % self.layer_num)
        print('[lulu] readout:%s' % self.readout)
        print('[lulu] atom_names:%s' % str(self.atom_names))
        print('[lulu] bond_names:%s' % str(self.bond_names))
        print('[lulu] bond_float_names:%s' % str(self.bond_float_names))

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(self, atom_bond_graph, van_graph):
        """
        Build the network.
        """
        node_hidden = self.init_atom_embedding(atom_bond_graph.node_feat)
        bond_embed = self.init_bond_embedding(atom_bond_graph.edge_feat)
        edge_hidden = bond_embed + self.init_bond_float_rbf(atom_bond_graph.edge_feat)

        van_edge_hidden = self.init_bond_van_float_rbf( van_graph.edge_feat )

        node_hidden_list = [node_hidden]
        van_node_hidden_list = [node_hidden]

        edge_hidden_list = [edge_hidden]
        van_edge_hidden_list = [van_edge_hidden]

        for layer_id in range(self.layer_num):
            node_hidden = self.atom_bond_block_list[layer_id](
                    atom_bond_graph,
                    node_hidden_list[layer_id],
                    edge_hidden_list[layer_id])
            
            
            van_node_hidden = self.atom_van_bond_list[layer_id](
                    van_graph,
                    van_node_hidden_list[layer_id],
                    van_edge_hidden_list[layer_id])
            
            cur_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph.edge_feat)
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](atom_bond_graph.edge_feat)

            cur_van_edge_hidden = self.atom_van_bond_rbf_list[layer_id](van_graph.edge_feat)
            
            node_hidden_list.append(node_hidden)
            van_node_hidden_list.append(van_node_hidden)

            edge_hidden_list.append(cur_edge_hidden)
            van_edge_hidden_list.append(cur_van_edge_hidden)
        
        node_repr = node_hidden_list[-1]
        van_node_repr = van_node_hidden_list[-1]

        edge_repr = edge_hidden_list[-1]
        van_edge_repr = van_edge_hidden_list[-1]

        graph_repr = self.graph_pool(atom_bond_graph, node_repr)
        van_graph_repr = self.graph_pool(van_graph, van_node_repr)

        node_repr = node_repr + van_node_repr
        # graph_repr = graph_repr + van_graph_repr

        return node_repr, edge_repr, graph_repr
