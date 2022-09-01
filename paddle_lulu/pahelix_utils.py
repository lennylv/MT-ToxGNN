import numpy as np
import os
import random
from pgl.utils.data import Dataloader
# from torch.utils.data import DataLoader
import json
from rdkit.Chem.Scaffolds import MurckoScaffold

def save_data_list_to_npz(data_list, npz_file):
    """
    Save a list of data to the npz file. Each data is a dict 
    of numpy ndarray.
    Args:   
        data_list(list): a list of data.
        npz_file(str): the npz file location.
    """
    keys = data_list[0].keys()
    merged_data = {}
    for key in keys:
        if len(np.array(data_list[0][key]).shape) == 0:
            lens = np.ones(len(data_list)).astype('int')
            values = np.array([data[key] for data in data_list])
            singular = 1
        else:
            lens = np.array([len(data[key]) for data in data_list])
            values = np.concatenate([data[key] for data in data_list], 0)
            singular = 0
        merged_data[key] = values
        merged_data[key + '.seq_len'] = lens
        merged_data[key + '.singular'] = singular
    np.savez_compressed(npz_file, **merged_data)

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    Args:
        smiles: smiles sequence
        include_chirality: Default=False
    
    Return: 
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def load_npz_to_data_list(npz_file):
    """
    Reload the data list save by ``save_data_list_to_npz``.
    Args:
        npz_file(str): the npz file location.
    Returns:
        a list of data where each data is a dict of numpy ndarray.
    """
    def _split_data(values, seq_lens, singular):
        res = []
        s = 0
        for l in seq_lens:
            if singular == 0:
                res.append(values[s: s + l])
            else:
                res.append(values[s])
            s += l
        return res

    merged_data = np.load(npz_file, allow_pickle=True)
    names = [name for name in merged_data.keys() 
            if not name.endswith('.seq_len') and not name.endswith('.singular')]
    data_dict = {}
    for name in names:
        data_dict[name] = _split_data(
                merged_data[name], 
                merged_data[name + '.seq_len'],
                merged_data[name + '.singular'])

    data_list = []
    n = len(data_dict[names[0]])
    for i in range(n):
        data = {name:data_dict[name][i] for name in names}
        data_list.append(data)
    return data_list


def get_part_files(data_path, trainer_id, trainer_num):
    """
    Split the files in data_path so that each trainer can train from different examples.
    """
    filenames = os.listdir(data_path)
    random.shuffle(filenames)
    part_filenames = []
    for (i, filename) in enumerate(filenames):
        if i % trainer_num == trainer_id:
            part_filenames.append(data_path + '/' + filename)
    return part_filenames

def mp_pool_map(list_input, func, num_workers):
    """list_output = [func(input) for input in list_input]"""
    from tqdm import tqdm
    class _CollateFn(object):
        def __init__(self, func):
            self.func = func
        def __call__(self, data_list):
            new_data_list = []
            for data in tqdm(data_list):
                index, input = data
                new_data_list.append((index, self.func(input)))
                # self.func(data)
            return new_data_list

    # print(1)
    # add index
    list_new_input = [(index, x) for index, x in enumerate(list_input)]
    
    data_gen = Dataloader(list_new_input, 
            batch_size=8200, 
            num_workers=4, 
            shuffle=False,
            collate_fn=_CollateFn(func))  
    

    list_output = []
    for sub_outputs in data_gen:
        list_output += sub_outputs
    # print(1)    
    list_output = sorted(list_output, key=lambda x: x[0])
    # remove index
    list_output = [x[1] for x in list_output]
    return list_output


def load_json_config(path):
    """tbd"""
    return json.load(open(path, 'r'))

class Splitter(object):
    """
    The abstract class of splitters which split up dataset into train/valid/test 
    subsets.
    """
    def __init__(self):
        super(Splitter, self).__init__()

class RandomSplitter(Splitter):
    """
    Random splitter.
    """
    def __init__(self):
        super(RandomSplitter, self).__init__()

    def split(self, 
            dataset, 
            frac_train=None, 
            frac_valid=None, 
            frac_test=None,
            seed=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split.
            frac_train(float): the fraction of data to be used for the train split.
            frac_valid(float): the fraction of data to be used for the valid split.
            frac_test(float): the fraction of data to be used for the test split.
            seed(int|None): the random seed.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        indices = list(range(N))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)

        train_dataset = dataset[indices[:train_cutoff]]
        valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
        test_dataset = dataset[indices[valid_cutoff:]]
        # np.save('./Free')
        if seed != None:
            np.save('./FreeSolvResult/' + str(seed) + '/train.npy', indices[:train_cutoff])
            np.save('./FreeSolvResult/' + str(seed) + '/valid.npy', indices[train_cutoff:valid_cutoff])
            np.save('./FreeSolvResult/' + str(seed) + '/test.npy', indices[valid_cutoff:])
        return train_dataset, valid_dataset, test_dataset

class ScaffoldSplitter(Splitter):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    
    Split dataset by Bemis-Murcko scaffolds
    """
    def __init__(self):
        super(ScaffoldSplitter, self).__init__()
    
    def split(self, 
            dataset, 
            frac_train=None, 
            frac_valid=None, 
            frac_test=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split. Make sure each element in
                the dataset has key "smiles" which will be used to calculate the 
                scaffold.
            frac_train(float): the fraction of data to be used for the train split.
            frac_valid(float): the fraction of data to be used for the valid split.
            frac_test(float): the fraction of data to be used for the test split.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = {}
        for i in range(N):
            scaffold = generate_scaffold(dataset[i]['smiles'], include_chirality=True)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)

        # sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

        # get train, valid test indices
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        # get train, valid test indices
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        return train_dataset, valid_dataset, test_dataset