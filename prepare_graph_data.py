import json

from Geo_lulu import GeoPredTransformFn_lulu
from paddle_lulu.Inmemory import InMemoryDataset
import pickle as pkl

# This file is used to generate graph data from smiles and 2d and 3d conformers.
# Then you can save your time when training

def smiles_to_dataset(smiles_list):
    """tbd"""
    data_list = smiles_list
    dataset = InMemoryDataset(data_list=data_list)
    return dataset

def load_json_config(path):
    """tbd"""
    return json.load(open(path, 'r'))

def make_files(smile_list, model_config, k):
    
    # pre_model.train()
    dataset = smiles_to_dataset(smiles_list=smile_list)

    print('Transform mini_batch', k, '/' , 1080000 // 1000)
    transform_fn = GeoPredTransformFn_lulu(model_config['pretrain_tasks'])
    dataset.transform(transform_fn, num_workers=1)

    f = open('./processed_data/' + str(k) + '.graphdata', 'wb')
    pkl.dump(dataset, f)
    f.close()
    # return np.mean(list_loss)

def main():
    f = open('./smiles_data.txt')
    lines = f.readlines()
    smiles_list_all = lines
    f.close()

    f = open('./2D_poses_cut.txt')
    poses = f.readlines()
    poses_all = poses
    f.close()

    f = open('./3D_poses_cut.txt')
    poses_3d = f.readlines()
    # print('read 3D poses...')
    # poses_3d_all = [eval(p.strip()) for p in tqdm(poses_3d[:20000])]
    poses_3d_all = poses_3d
    f.close()

    model_config = load_json_config('./config/pretrain_gem.json')

    for epoch in range(0,1):

        # print('Trainging...')
        size = 1000
        for i in range(0, len(smiles_list_all), size):
            smiles_list = [eval(l.strip()) for l in smiles_list_all[i:i + size]]
            # print('read 2D poses...')
            poses = [eval(l.strip()) for l in poses_all[i:i + size]]
            # print('read 3D poses...')
            poses_3d = [eval(l.strip()) for l in poses_3d_all[i:i + size]]

            smiles_list = [(smiles_list[j][0], smiles_list[j][1], poses[j], poses_3d[j]) for j in range(len(smiles_list))]

            make_files(smiles_list, model_config, i//size + 1)

if __name__ == '__main__':
    main()