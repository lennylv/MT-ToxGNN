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
        self.hidden = model_config.get('hidden_dim', 512)
        self.compound_encoder = compound_encoder
        self.norm = nn.LayerNorm(compound_encoder.graph_dim)
        # self.mlp = MLP(
        #         # model_config['layer_num'],
        #         4,
        #         in_size=compound_encoder.graph_dim,
        #         hidden_size=512,
        #         out_size=model_config['hidden_size'],
        #         act=model_config['act'],
        #         dropout_rate=model_config['dropout_rate'])

        self.seq = nn.Sequential(
        nn.Linear(compound_encoder.graph_dim, self.hidden),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(self.hidden , self.hidden // 2),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(self.hidden // 2, 1),
        # nn.LeakyReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(self.hidden // 4, 1)
        )

    def forward(self, atom_bond_graphs, bond_angle_graphs, task='tox'):
        """
        Define the forward function,set the parameter layer options.compound_encoder 
        creates a graph data holders that attributes and features in the graph.
        Returns:
            pred: the model prediction.
        """
        node_repr, edge_repr, graph_repr = self.compound_encoder(atom_bond_graphs, bond_angle_graphs)
        graph_repr = self.norm(graph_repr)
        pred = self.seq(graph_repr)

        return pred