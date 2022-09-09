from functools import wraps
from textwrap import wrap
import numpy as np
from sklearn.metrics import pairwise_distances
from rdkit import Chem
from rdkit.Chem import AllChem
from CompoundKit_lulu import CompoundKit, Compound3DKit
import pgl
import random

def get_pretrain_bond_angle(edges, atom_poses):
    """tbd"""
    def _get_angle(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
        vec2 = vec2 / (norm2 + 1e-5)
        angle = np.arccos(np.dot(vec1, vec2))
        return angle
    def _add_item(
            node_i_indices, node_j_indices, node_k_indices, bond_angles, 
            node_i_index, node_j_index, node_k_index):
        node_i_indices += [node_i_index, node_k_index]
        node_j_indices += [node_j_index, node_j_index]
        node_k_indices += [node_k_index, node_i_index]
        pos_i = atom_poses[node_i_index]
        pos_j = atom_poses[node_j_index]
        pos_k = atom_poses[node_k_index]
        angle = _get_angle(pos_i - pos_j, pos_k - pos_j)
        bond_angles += [angle, angle]

    E = len(edges)
    node_i_indices = []
    node_j_indices = []
    node_k_indices = []
    bond_angles = []
    for edge_i in range(E - 1):
        for edge_j in range(edge_i + 1, E):
            a0, a1 = edges[edge_i]
            b0, b1 = edges[edge_j]
            if a0 == b0 and a1 == b1:
                continue
            if a0 == b1 and a1 == b0:
                continue
            if a0 == b0:
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a1, a0, b1)
            if a0 == b1:
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a1, a0, b0)
            if a1 == b0:
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a0, a1, b1)
            if a1 == b1:
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a0, a1, b0)
    node_ijk = np.array([node_i_indices, node_j_indices, node_k_indices])
    uniq_node_ijk, uniq_index = np.unique(node_ijk, return_index=True, axis=1)
    node_i_indices, node_j_indices, node_k_indices = uniq_node_ijk
    bond_angles = np.array(bond_angles)[uniq_index]
    return [node_i_indices, node_j_indices, node_k_indices, bond_angles]

def mol_to_graph_data(mol):
    """
    mol_to_graph_data
    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = [
        "atomic_num", "chiral_tag", "degree", "explicit_valence",
        "formal_charge", "hybridization", "implicit_valence",
        "is_aromatic", "total_numHs",
    ]

    # atom_id_names = [
    # "atomic_num", 
    # "chiral_tag",
    # "degree",
    # "explicit_valence",
    # "formal_charge",
    # "hybridization",
    # "implicit_valence", 
    # "is_aromatic", 
    # "total_numHs",
    # "num_radical_e",
    # "atom_is_in_ring",
    # "valence_out_shell",
    # "in_num_ring_with_size3",
    # "in_num_ring_with_size4",
    # "in_num_ring_with_size5",
    # "in_num_ring_with_size6",
    # "in_num_ring_with_size7",
    # "in_num_ring_with_size8"
    # ]

    bond_id_names = [
        "bond_dir", "bond_type", "is_in_ring",
    ]
    
    data = {}
    for name in atom_id_names:
        data[name] = []
    data['mass'] = []
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    ### atom features
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return None
        for name in atom_id_names:
            data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)  # 0: OOV
        data['mass'].append(CompoundKit.get_atom_value(atom, 'mass') * 0.01)

    ### bond features
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name) + 1   # 0: OOV
            data[name] += [bond_feature_id] * 2

    ### self loop (+2)
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = CompoundKit.get_bond_feature_size(name) + 2   # N + 2: self loop
        data[name] += [bond_feature_id] * N

    ### check whether edge exists
    if len(data['edges']) == 0: # mol has no bonds
        for name in bond_id_names:
            data[name] = np.zeros((0,), dtype="int64")
        data['edges'] = np.zeros((0, 2), dtype="int64")

    ### make ndarray and check length
    for name in atom_id_names:
        data[name] = np.array(data[name], 'int64')
    data['mass'] = np.array(data['mass'], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    ### morgan fingerprint
    # data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    # # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    # data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    # data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')

    return data



def mol_to_geognn_graph_data(mol, atom_poses, atom_poses_3D = None):
    """
    mol: rdkit molecule
    dir_type: direction type for bond_angle grpah
    """
    if len(mol.GetAtoms()) == 0:
        return None

    data = mol_to_graph_data(mol)

    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])

    if atom_poses_3D != None:
        data['atom_pos_3D'] = np.array(atom_poses_3D, 'float32')
        data['bond_length_3D'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos_3D'])

    # BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
    #         Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'])
    # data['BondAngleGraph_edges'] = BondAngleGraph_edges
    # data['bond_angle'] = np.array(bond_angles, 'float32')

    return data


def mol_to_geognn_graph_data_2d(smiles, mol, atom_poses, atom_poses_3D):
    """tbd"""
    # if len(mol.GetAtoms()) <= 400:
    #     # mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    #     if dataset == 'qm9':
    #         f = open('./qm9/' + smiles + '.txt')
    #         atom_poses = eval(f.read())
    #         f.close()
    #     elif dataset == 'drugs':
    #         f = open('./drugs_geom/' + str(smiles) + '.txt')
    #         lines = f.readlines()
    #         atom_poses = eval(''.join(lines[-2:]))
    #         f.close()
    #         # atom_poses = Compound3DKit.get_qm9_atoms_poses(mol)
    # else:
    #     atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    return mol_to_geognn_graph_data(mol, atom_poses, atom_poses_3D)

def lorentz_func(van_der_waals_r1, van_der_waals_r2, atiomic_distance):
    kernal_tau = 0.5
    kernal_parameter = 10.0
    eta = kernal_tau * (van_der_waals_r1 + van_der_waals_r2)
    phi = 1 / (1 + atiomic_distance/eta) ** kernal_parameter
    return np.round(phi, 5)

def van_der_edges( mol, data, ratio ):
    cutoff = 12
    sigma = 0  # mean(std(ri),std(rk)) in dataset

    # Atomic radii
    atomic_r = {
        'H': 0.53,
        'C': 0.67,
        'N': 0.56,
        'O': 0.48,
        'F': 0.42,
        'P': 0.98,
        'S': 0.87,
        'Cl': 0.79,
        'Br': 0.94,
        'I': 1.15,
        'B':0.87,
        'Bi':1.43,
        'Si':1.11,
        'As':1.14,
        'Al':1.18,
        'Hg':1.71,
        'Se':1.03,
        'Na':1.90,
        'Ca':1.94
    }

    # van der Waals radii
    van_der_waals_r = {
        'H': 1.2,
        'C': 1.77,
        'N': 1.66,
        'O': 1.5,
        'F': 1.46,
        'P': 1.9,
        'S': 1.89,
        'Cl': 1.82,
        'Br': 1.86,
        'I': 2.04,
        'B': 1.95,
        'Bi':2.31,
        'Si':2.11,
        'As':2.05,
        'Al':2.24,
        'Hg':2.07,
        'Se':1.90,
        'Na':2.27,
        'Ca':2.31
    }

    atoms = mol.GetAtoms()
    r = [atomic_r[atom.GetSymbol()] for atom in atoms]
    van_r_list = [van_der_waals_r[atom.GetSymbol()] for atom in atoms]

    min_dis = [[r[i]+r[j]+sigma for i in range(len(r))] for j in range(len(r))]
    min_dis = np.array(min_dis)

    # 2D input van der walls
    dist_matrix = pairwise_distances(data['atom_pos'])
    dist_matrix[dist_matrix < min_dis] = 0
    dist_matrix[dist_matrix > cutoff] = 0

    # 3D predict van der walls
    if 'atom_pos_3D' in data.keys():
        dist_matrix_3d = pairwise_distances(data['atom_pos_3D'])
        dist_matrix_3d[dist_matrix < min_dis] = 0
        dist_matrix_3d[dist_matrix > cutoff] = 0    

    van_bond_edge = np.argwhere(dist_matrix!=0)

    copy_edge = [tuple(e) for e in data['edges']]
    van_bond_edge = [b for b in van_bond_edge if (b[0], b[1]) not in copy_edge]
    left_ratio = ratio
    # left_number = len(van_bond_edge) * left_ratio
    # left_number = int(left_number)
    # random.shuffle(van_bond_edge)
    # count_van_dic = {}
    van_bond_edge_copy = []
    # for van_pair in van_bond_edge:
    for j in range(len(atoms)):
        temp = [i for i in van_bond_edge if i[0] == j]
        # count_van_dic[j] = len(temp)
        left_number = int(len(temp) * left_ratio)
        temp = temp[:left_number]
        van_bond_edge_copy.extend(temp)
    # print(van_bond_edge_copy)

    
    # van_bond_edge_copy = [tuple(e) for e in van_bond_edge_copy]
    # for vd in van_bond_edge_copy:
    #     if (vd[1],vd[0]) in van_bond_edge_copy:
    #         continue
    #     else:
    #         van_bond_edge_copy.append((vd[1],vd[0]))

        
    van_bond_edge = van_bond_edge_copy

    van_bond_edge = np.array(van_bond_edge, 'int64')
    # van_bond_edge = [tuple(b) for b in van_bond_edge]

    van_der_walls = [lorentz_func(van_r_list[b_p[0]], van_r_list[b_p[1]], dist_matrix[b_p[0]][b_p[1]]) for b_p in van_bond_edge]

    if 'atom_pos_3D' in data.keys():
        van_der_walls_3D = [lorentz_func(van_r_list[b_p[0]], van_r_list[b_p[1]], dist_matrix_3d[b_p[0]][b_p[1]]) for b_p in van_bond_edge]
        data['van_der_walls_3D'] = np.array(van_der_walls_3D, 'float32')

    if len(van_bond_edge) == 0:
        van_bond_edge = [(i, i) for i in range(len(r))]
        van_der_walls = [0. for i in range(len(r))]
        
    data['van_edges'] = van_bond_edge
    data['van_der_walls'] = np.array(van_der_walls, 'float32') 

    # data['van_der_walls_3D'] = np.array(van_der_walls_3D, 'float32') 
    return data

class GeoPredTransformFn_lulu(object):
    """Gen features for downstream model"""
    def __init__(self, pretrain_tasks):
        self.pretrain_tasks = pretrain_tasks
        # self.mask_ratio = mask_ratio

    def prepare_pretrain_task(self, data):
        """
        prepare data for pretrain task
        """
        node_i, node_j, node_k, bond_angles = \
                get_pretrain_bond_angle(data['edges'], data['atom_pos'])
        
        data['Ba_node_i'] = node_i
        data['Ba_node_j'] = node_j
        data['Ba_node_k'] = node_k
        data['Ba_bond_angle'] = bond_angles

        data['Bl_node_i'] = np.array(data['edges'][:, 0])
        data['Bl_node_j'] = np.array(data['edges'][:, 1])
        data['Bl_bond_length'] = np.array(data['bond_length_3D'])

        data['Av_node_i'] = np.array(data['van_edges'][:, 0])
        data['Av_node_j'] = np.array(data['van_edges'][:, 1])
        data['Av_walls'] = np.array(data['van_der_walls_3D'])

        n = len(data['atom_pos'])
        dist_matrix = pairwise_distances(data['atom_pos_3D'])
        indice = np.repeat(np.arange(n).reshape([-1, 1]), n, axis=1)
        data['Ad_node_i'] = indice.reshape([-1, 1])
        data['Ad_node_j'] = indice.T.reshape([-1, 1])
        data['Ad_atom_dist'] = dist_matrix.reshape([-1, 1])

        #
        # data['Aa_atom_ajacency'] = data['adjacency'].reshape([-1, 1])
        
        return data
    # @wraps
    def __call__(self, raw_data):
        """
        Gen features according to raw data and return a single graph data.
        Args:
            raw_data: It contains smiles and label,we convert smiles 
            to mol by rdkit,then convert mol to graph data.
        Returns:
            data: It contains reshape label and smiles.
        """
        # smiles = raw_data[0]
        smiles = raw_data[1]
        atom_poses = raw_data[2]
        atom_poses_3D = raw_data[3]
        ratio = raw_data[4]
        # print('smiles', smiles)
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        if raw_data[0] == 'q':
            data = mol_to_geognn_graph_data_2d(smiles, mol, atom_poses, atom_poses_3D)
        # elif raw_data[1][0] == 'd':
        else:
            # print(raw_data[0])
            data = mol_to_geognn_graph_data_2d(raw_data[0], mol, atom_poses, atom_poses_3D)
        
        data = van_der_edges( mol, data ,ratio)
        data['smiles'] = smiles
        data = self.prepare_pretrain_task(data)

        return data

class DownstreamTransformFn(object):
    """Gen features for downstream model"""
    def __init__(self, is_inference=False):
        self.is_inference = is_inference

    def __call__(self, raw_data):
        """
        Gen features according to raw data and return a single graph data.
        Args:
            raw_data: It contains smiles and label,we convert smiles 
            to mol by rdkit,then convert mol to graph data.
        Returns:
            data: It contains reshape label and smiles.
        """
        #  raw_data: mol, atom_poses, label
        # smiles = raw_data['smiles']
        # print(smiles)
        # mol = AllChem.MolFromSmiles(smiles)

        mol = raw_data[0]
        smiles = Chem.MolToSmiles(mol)
        # print(smiles)
        atom_poses = raw_data[1]
        label = raw_data[2]
        ratio = raw_data[3]
        if mol is None:
            return None
        data = mol_to_geognn_graph_data_2d(None, mol, atom_poses, None)
        data = van_der_edges( mol, data, ratio )
        if not self.is_inference:
            # data['label'] = raw_data['label'].reshape([-1])
            data['label'] = label
        data['smiles'] = smiles
        return data


