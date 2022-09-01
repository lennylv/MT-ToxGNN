import numpy as np

import paddle
import paddle.nn as nn
import pgl
from paddle_models.GNN_blocks import MeanPool
# from paddle_models.MolEncoder import GeoGNNBlock

from CompoundKit_lulu import CompoundKit
# from pahelix.networks.basic_block import RBF

class RBF(nn.Layer):
    """
    Radial Basis Function
    """
    def __init__(self, centers, gamma, dtype='float32'):
        super(RBF, self).__init__()
        self.centers = paddle.reshape(paddle.to_tensor(centers, dtype=dtype), [1, -1])
        self.gamma = gamma
    
    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = paddle.reshape(x, [-1, 1])
        return paddle.exp(-self.gamma * paddle.square(x - self.centers))

class AtomEmbedding(nn.Layer):
    """
    Atom Encoder
    """
    def __init__(self, atom_names, embed_dim):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names
        
        self.embed_list = nn.LayerList()
        for name in self.atom_names:
            embed = nn.Embedding(
                    CompoundKit.get_atom_feature_size(name) + 5,
                    embed_dim, 
                    weight_attr=nn.initializer.XavierUniform())
            self.embed_list.append(embed)

    def forward(self, node_features):
        """
        Args: 
            node_features(dict of tensor): node features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_names):
            out_embed += self.embed_list[i](node_features[name])
        return out_embed


class AtomFloatEmbedding(nn.Layer):
    """
    Atom Float Encoder
    """
    def __init__(self, atom_float_names, embed_dim, rbf_params=None):
        super(AtomFloatEmbedding, self).__init__()
        self.atom_float_names = atom_float_names
        
        if rbf_params is None:
            self.rbf_params = {
                'van_der_waals_radis': (np.arange(1, 3, 0.2), 10.0),   # (centers, gamma)
                'partial_charge': (np.arange(-1, 4, 0.25), 10.0),   # (centers, gamma)
                'mass': (np.arange(0, 2, 0.1), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.LayerList()
        self.rbf_list = nn.LayerList()
        for name in self.atom_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, feats):
        """
        Args: 
            feats(dict of tensor): node float features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_float_names):
            x = feats[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class BondEmbedding(nn.Layer):
    """
    Bond Encoder
    """
    def __init__(self, bond_names, embed_dim):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names
        
        self.embed_list = nn.LayerList()
        for name in self.bond_names:
            embed = nn.Embedding(
                    CompoundKit.get_bond_feature_size(name) + 5,
                    embed_dim, 
                    weight_attr=nn.initializer.XavierUniform())
            self.embed_list.append(embed)

    def forward(self, edge_features):
        """
        Args: 
            edge_features(dict of tensor): edge features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_names):
            out_embed += self.embed_list[i](edge_features[name])
        return out_embed


class BondFloatRBF(nn.Layer):
    """
    Bond Float Encoder using Radial Basis Functions
    """
    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (np.arange(0, 2, 0.1), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
        
        self.linear_list = nn.LayerList()
        self.rbf_list = nn.LayerList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args: 
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed

class VanBondFloatRBF(nn.Layer):
    """
    Van der walls Float Encoder using Radial Basis Functions
    """
    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(VanBondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'van_der_walls': (np.arange(1, 3, 0.2), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
        
        self.linear_list = nn.LayerList()
        self.rbf_list = nn.LayerList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed

class BondAngleFloatRBF(nn.Layer):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """
    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (np.arange(0, np.pi, 0.1), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
        
        self.linear_list = nn.LayerList()
        self.rbf_list = nn.LayerList()
        for name in self.bond_angle_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, bond_angle_float_features):
        """
        Args: 
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            x = bond_angle_float_features[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed

class GeoGNNModel(nn.Layer):
    """
    The GeoGNN Model used in GEM.
    Args:
        model_config(dict): a dict of model configurations.
    """
    def __init__(self, model_config={}):
        super(GeoGNNModel, self).__init__()

        self.embed_dim = model_config.get('embed_dim', 32)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        self.layer_num = model_config.get('layer_num', 8)
        self.readout = model_config.get('readout', 'mean')

        self.atom_names = model_config['atom_names']
        self.bond_names = model_config['bond_names']
        self.bond_float_names = model_config['bond_float_names']
        self.bond_angle_float_names = model_config['bond_angle_float_names']

        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim)
        
        self.bond_embedding_list = nn.LayerList()
        self.bond_float_rbf_list = nn.LayerList()
        self.bond_angle_float_rbf_list = nn.LayerList()
        self.atom_bond_block_list = nn.LayerList()
        self.bond_angle_block_list = nn.LayerList()
        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(
                    BondEmbedding(self.bond_names, self.embed_dim))
            self.bond_float_rbf_list.append(
                    BondFloatRBF(self.bond_float_names, self.embed_dim))
            self.bond_angle_float_rbf_list.append(
                    BondAngleFloatRBF(self.bond_angle_float_names, self.embed_dim))
            self.atom_bond_block_list.append(
                    GeoGNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
            self.bond_angle_block_list.append(
                    GeoGNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
        
        # TODO: use self-implemented MeanPool due to pgl bug.
        if self.readout == 'mean':
            self.graph_pool = MeanPool()
        else:
            self.graph_pool = pgl.nn.GraphPool(pool_type=self.readout)

        print('[GeoGNNModel] embed_dim:%s' % self.embed_dim)
        print('[GeoGNNModel] dropout_rate:%s' % self.dropout_rate)
        print('[GeoGNNModel] layer_num:%s' % self.layer_num)
        print('[GeoGNNModel] readout:%s' % self.readout)
        print('[GeoGNNModel] atom_names:%s' % str(self.atom_names))
        print('[GeoGNNModel] bond_names:%s' % str(self.bond_names))
        print('[GeoGNNModel] bond_float_names:%s' % str(self.bond_float_names))
        print('[GeoGNNModel] bond_angle_float_names:%s' % str(self.bond_angle_float_names))

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(self, atom_bond_graph, bond_angle_graph):
        """
        Build the network.
        """
        node_hidden = self.init_atom_embedding(atom_bond_graph.node_feat)
        bond_embed = self.init_bond_embedding(atom_bond_graph.edge_feat)
        edge_hidden = bond_embed + self.init_bond_float_rbf(atom_bond_graph.edge_feat)

        node_hidden_list = [node_hidden]
        edge_hidden_list = [edge_hidden]
        for layer_id in range(self.layer_num):
            node_hidden = self.atom_bond_block_list[layer_id](
                    atom_bond_graph,
                    node_hidden_list[layer_id],
                    edge_hidden_list[layer_id])
            
            cur_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph.edge_feat)
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](atom_bond_graph.edge_feat)
            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angle_graph.edge_feat)
            edge_hidden = self.bond_angle_block_list[layer_id](
                    bond_angle_graph,
                    cur_edge_hidden,
                    cur_angle_hidden)
            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)
        
        node_repr = node_hidden_list[-1]
        edge_repr = edge_hidden_list[-1]
        graph_repr = self.graph_pool(atom_bond_graph, node_repr)
        return node_repr, edge_repr, graph_repr