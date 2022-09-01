import pgl
import numpy as np
from copy import deepcopy
import hashlib


def md5_hash(string):
    """tbd"""
    md5 = hashlib.md5(string.encode('utf-8')).hexdigest()
    return int(md5, 16)

def mask_context_of_geognn_graph(
        g, 
        # superedge_g,
        van_g,
        target_atom_indices=None, 
        mask_ratio=None, 
        mask_value=0, 
        subgraph_num=None,
        version='gem'):
    """tbd"""
    def get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices):
        """tbd"""
        atomic_num = g.node_feat['atomic_num'].flatten()
        bond_type = g.edge_feat['bond_type'].flatten()
        subgraph_str = 'A' + str(atomic_num[atom_index])
        subgraph_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
        subgraph_str += 'E' + ':'.join([str(x) for x in np.sort(bond_type[nei_bond_indices])])
        return subgraph_str
    
    def get_subgraph_str_van(g, atom_index, nei_atom_indices, nei_bond_indices):
        """tbd"""
        atomic_num = g.node_feat['atomic_num'].flatten()
        # bond_type = g.edge_feat['van_der_walls'].flatten()
        edge = np.array([int(str(e[0])+str(e[1])) for e in g.edges])
        subgraph_str = 'A' + str(atomic_num[atom_index])
        subgraph_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
        subgraph_str += 'E' + ':'.join([str(x) for x in np.sort(edge[nei_bond_indices])])
        return subgraph_str

    g = deepcopy(g)
    N = g.num_nodes
    E = g.num_edges
    full_atom_indices = np.arange(N)
    full_bond_indices = np.arange(E)
    

    van_g = deepcopy(van_g)
    E_van = van_g.num_edges
    full_van_bond_indices = np.arange(E_van)
    ############################

    if target_atom_indices is None:
        masked_size = max(1, int(N * mask_ratio))   # at least 1 atom will be selected.
        target_atom_indices = np.random.choice(full_atom_indices, size=masked_size, replace=False)
    target_labels = []
    Cm_node_i = []
    masked_bond_indices = []


    target_labels_van = []
    masked_bond_indices_van = []
    ############################

    for atom_index in target_atom_indices:
        left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == atom_index]
        right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == atom_index]
        nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)

        left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
        right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
        nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)

        left_nei_van_bond_indices = full_van_bond_indices[van_g.edges[:, 0] == atom_index]
        right_nei_van_bond_indices = full_van_bond_indices[van_g.edges[:, 1] == atom_index]
        nei_van_bond_indices = np.append( left_nei_van_bond_indices, right_nei_van_bond_indices)

        left_nei_van_atom_indices = van_g.edges[left_nei_van_bond_indices, 1]
        right_nei_van_atom_indices = van_g.edges[right_nei_van_bond_indices, 0]
        nei_van_atoms_indices = np.append( left_nei_van_atom_indices, right_nei_van_atom_indices )

        if version == 'gem':
            subgraph_str = get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices)
            subgraph_id = md5_hash(subgraph_str) % subgraph_num
            target_label = subgraph_id

            subgraph_str_van = get_subgraph_str_van(van_g, atom_index, nei_van_atoms_indices, nei_van_bond_indices)
            subgraph_id_van = md5_hash(subgraph_str_van)
            target_label_van = int(str(subgraph_id_van)[:4], 16)
        else:
            raise ValueError(version)
        
        target_labels.append(target_label)
        Cm_node_i.append([atom_index])
        Cm_node_i.append(nei_atom_indices)
        masked_bond_indices.append(nei_bond_indices)

        # lulu
        target_labels_van.append(target_label_van)
        masked_bond_indices_van.append(nei_van_bond_indices)

        #########################################################


    target_atom_indices = np.array(target_atom_indices)
    target_labels = np.array(target_labels)
    Cm_node_i = np.concatenate(Cm_node_i, 0)
    masked_bond_indices = np.concatenate(masked_bond_indices, 0)

    # lulu 
    target_labels_van = np.array( target_labels_van )
    masked_bond_indices_van = np.concatenate(masked_bond_indices_van, 0)
    #########################################################

    for name in g.node_feat:
        g.node_feat[name][Cm_node_i] = mask_value
        # lulu 
        van_g.node_feat[name][Cm_node_i] = mask_value
        #######################################
    for name in g.edge_feat:
        g.edge_feat[name][masked_bond_indices] = mask_value
    
    # lulu
    for name in van_g.edge_feat:
        van_g.edge_feat[name][masked_bond_indices_van] = mask_value
    ###########################################################

    return [g, target_atom_indices, target_labels, van_g, target_labels_van]


class GeoPredCollateFn_lulu(object):
    """tbd"""
    def __init__(self,
             atom_names,
             bond_names,
             bond_float_names,
             bond_angle_float_names,
             pretrain_tasks,
             mask_ratio,
             atom_van_bond_names,
             Cm_vocab
             ):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.pretrain_tasks = pretrain_tasks
        self.mask_ratio = mask_ratio
        self.Cm_vocab = Cm_vocab
        self.bond_angle_float_names = bond_angle_float_names
        self.atom_van_bond_names = atom_van_bond_names
        
    def _flat_shapes(self, d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])

    def __call__(self, batch_data_list):
        """tbd"""
        atom_bond_graph_list = []
        atom_van_graph_list = []


        masked_atom_bond_graph_list = []
        masked_atom_van_graph_list = []

        Cm_node_i = []
        Cm_context_id = []
        Cm_context_id_van = []

        Bl_node_i = []
        Bl_node_j = []
        Bl_bond_length = []

        Av_node_i = []
        Av_node_j = []
        Av_van = []

        Ad_node_i = []
        Ad_node_j = []
        Ad_atom_dist = []

        node_count = 0
        for data in batch_data_list:
            N = len(data[self.atom_names[0]])
            # E = len(data['edges'])
            ab_g = pgl.graph.Graph(num_nodes=N,
                    edges = data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names})
            av_g = pgl.graph.Graph(num_nodes=N,
                    edges = data['van_edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.atom_van_bond_names})
            # masked_ab_g, masked_ba_g, mask_node_i, context_id = mask_context_of_geognn_graph(
                    # ab_g, ba_g, mask_ratio=self.mask_ratio, subgraph_num=self.Cm_vocab)
            atom_bond_graph_list.append(ab_g)

            # lulu
            masked_ab_g, mask_node_i, context_id, masked_av_g, context_van_id = mask_context_of_geognn_graph(
                    ab_g, av_g, mask_ratio=self.mask_ratio, subgraph_num=self.Cm_vocab)   
            atom_van_graph_list.append(av_g)
            ##################################################

            masked_atom_bond_graph_list.append(masked_ab_g)
            # lulu
            masked_atom_van_graph_list.append(masked_av_g)
            ##################################################
            if 'Cm' in self.pretrain_tasks:
                Cm_node_i.append(mask_node_i + node_count)
                Cm_context_id.append(context_id)

                Cm_context_id_van.append(context_van_id)


            if 'Blr' in self.pretrain_tasks:
                Bl_node_i.append(data['Bl_node_i'] + node_count)
                Bl_node_j.append(data['Bl_node_j'] + node_count)
                Bl_bond_length.append(data['Bl_bond_length'])

            if 'van' in self.pretrain_tasks:
                Av_node_i.append(data['Av_node_i'] + node_count)
                Av_node_j.append(data['Av_node_j'] + node_count)
                Av_van.append(data['Av_walls'])

            if 'Adc' in self.pretrain_tasks:
                Ad_node_i.append(data['Ad_node_i'] + node_count)
                Ad_node_j.append(data['Ad_node_j'] + node_count)
                Ad_atom_dist.append(data['Ad_atom_dist'])

            node_count += N

        graph_dict = {}
        feed_dict = {}
        
        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        graph_dict['atom_bond_graph'] = atom_bond_graph

        atom_van_graph = pgl.Graph.batch(atom_van_graph_list)
        self._flat_shapes(atom_van_graph.node_feat)
        self._flat_shapes(atom_van_graph.edge_feat)
        graph_dict['atom_van_graph'] = atom_van_graph


        masked_atom_bond_graph = pgl.Graph.batch(masked_atom_bond_graph_list)
        self._flat_shapes(masked_atom_bond_graph.node_feat)
        self._flat_shapes(masked_atom_bond_graph.edge_feat)
        graph_dict['masked_atom_bond_graph'] = masked_atom_bond_graph

        # lulu
        masked_atom_van_bond_graph = pgl.Graph.batch(masked_atom_van_graph_list)
        self._flat_shapes(masked_atom_van_bond_graph.node_feat)
        self._flat_shapes(masked_atom_van_bond_graph.edge_feat)
        graph_dict['masked_atom_van_bond_graph'] = masked_atom_van_bond_graph

        if 'Cm' in self.pretrain_tasks:
            feed_dict['Cm_node_i'] = np.concatenate(Cm_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Cm_context_id'] = np.concatenate(Cm_context_id, 0).reshape(-1, 1).astype('int64')

            feed_dict['Cm_context_id_van'] = np.concatenate(Cm_context_id_van, 0).reshape(-1, 1).astype('int64')

        if 'Blr' in self.pretrain_tasks:
            feed_dict['Bl_node_i'] = np.concatenate(Bl_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Bl_node_j'] = np.concatenate(Bl_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Bl_bond_length'] = np.concatenate(Bl_bond_length, 0).reshape(-1, 1).astype('float32')

        if 'van' in self.pretrain_tasks:
            feed_dict['Av_node_i'] = np.concatenate(Av_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Av_node_j'] = np.concatenate(Av_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Av_walls'] = np.concatenate(Av_van, 0).reshape(-1, 1).astype('float32')

        if 'Adc' in self.pretrain_tasks:
            feed_dict['Ad_node_i'] = np.concatenate(Ad_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Ad_node_j'] = np.concatenate(Ad_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Ad_atom_dist'] = np.concatenate(Ad_atom_dist, 0).reshape(-1, 1).astype('float32')
            # feed_dict['Aa_atom_adj'] = np.concatenate(Aa_atom_adj, 0).reshape(-1, 1).astype('float32')

        return graph_dict, feed_dict

class DownstreamCollateFn(object):
    """CollateFn for downstream model"""
    def __init__(self, 
            atom_names, 
            bond_names, 
            bond_float_names,
            bond_angle_float_names,
            task_type,
            atom_van_bond_names,
            is_inference=False):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names
        self.atom_van_bond_names = atom_van_bond_names

        self.task_type = task_type
        self.is_inference = is_inference

    def _flat_shapes(self, d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])
    
    def __call__(self, data_list):
        """
        Collate features about a sublist of graph data and return join_graph, 
        masked_node_indice and masked_node_labels.
        Args:
            data_list : the graph data in gen_features.for data in data_list,
            create node features and edge features according to pgl graph,and then 
            use graph wrapper to feed join graph, then the label can be arrayed to batch label.
        Returns:
            The batch data contains finetune label and valid,which are 
            collected from batch_label and batch_valid.  
        """
        atom_bond_graph_list = []
        atom_van_graph_list = []

        label_list = []
        for data in data_list:
            ab_g = pgl.Graph(
                    num_nodes=len(data[self.atom_names[0]]),
                    edges=data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names})


            av_g = pgl.graph.Graph(num_nodes=len(data[self.atom_names[0]]),
                    edges = data['van_edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.atom_van_bond_names})

            atom_bond_graph_list.append(ab_g)
            atom_van_graph_list.append(av_g)

            if not self.is_inference:
                label_list.append(data['label'])

        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        atom_van_graph = pgl.Graph.batch(atom_van_graph_list)
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        self._flat_shapes(atom_van_graph.node_feat)
        self._flat_shapes(atom_van_graph.edge_feat)

        if not self.is_inference:
            if self.task_type == 'class':
                # print(label_list)
                labels = np.array(label_list).astype(float)
                # print(labels)
                # label: -1 -> 0, 1 -> 1
                # labels = ((labels + 1.0) / 2)
                # valids = (labels != 0.5)
                return atom_bond_graph, atom_van_graph, labels
            else:
                labels = np.array(label_list, 'float32')
                return atom_bond_graph, atom_van_graph, labels
        else:
            return atom_bond_graph, atom_van_graph