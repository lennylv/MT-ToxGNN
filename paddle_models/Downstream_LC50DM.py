import paddle
import paddle.nn as nn
import pgl

from paddle_models.PredictionMol import MLP


class DownstreamModel(nn.Layer):
    """
    Docstring for DownstreamModel,it is an supervised 
    GNN model which predicts the tasks shown in num_tasks and so on.
    """
    def __init__(self, model_config, compound_encoder):
        super(DownstreamModel, self).__init__()
        self.task_type = model_config['task_type']
        self.num_tasks = 1

        self.compound_encoder = compound_encoder
        self.norm = nn.LayerNorm(compound_encoder.graph_dim)
        self.mlp = MLP(
                # model_config['layer_num'],
                4,
                in_size=compound_encoder.graph_dim,
                hidden_size=512,
                out_size=256,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate'])

        # self.mlp2 = MLP(
        #         # model_config['layer_num'],
        #         4,
        #         in_size=model_config['hidden_size'],
        #         hidden_size=512,
        #         out_size=model_config['hidden_size'],
        #         act=model_config['act'],
        #         dropout_rate=model_config['dropout_rate'])

        self.act = nn.ReLU()
        self.fc1 = MLP(
                # model_config['layer_num'],
                2,
                in_size=256,
                hidden_size=256,
                out_size=1,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate'])

        self.fc2 = MLP(
                # model_config['layer_num'],
                2,
                in_size=256,
                hidden_size=256,
                out_size=1,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate'])

        self.fc3 = MLP(
                # model_config['layer_num'],
                2,
                in_size=256,
                hidden_size=256,
                out_size=1,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate'])

        self.fc4 = MLP(
                # model_config['layer_num'],
                2,
                in_size=256,
                hidden_size=256,
                out_size=1,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate'])

        if self.task_type == 'class':
            self.out_act = nn.Sigmoid()

    def forward(self, atom_bond_graphs, bond_angle_graphs, task='tox'):
        """
        Define the forward function,set the parameter layer options.compound_encoder 
        creates a graph data holders that attributes and features in the graph.
        Returns:
            pred: the model prediction.
        """
        node_repr, edge_repr, graph_repr = self.compound_encoder(atom_bond_graphs, bond_angle_graphs)
        graph_repr = self.norm(graph_repr)
        pred = self.mlp(graph_repr)
        # pred = self.mlp2(pred)
        if self.task_type == 'class':
            pred = self.out_act(pred)
        # pred = self.act(pred)

        if task == 'LC50DM':
            pred = self.fc1(pred)
        elif task == 'LC50':
            pred = self.fc2(pred)
        elif task == 'IGC50':
            pred = self.fc3(pred)
        elif task == 'logP':
            pred = self.fc1(pred)
        else:
            pred = self.fc4(pred)
        return pred