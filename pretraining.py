import numpy as np
import os
import time
import sys
# import torch
# from pretrain_utils import *
import json
import pickle as pkl
import paddle
# from xgboost import train

from Predcollatefn import PredCollateFn
from Geo_lulu import PredTransformFn
from paddle_lulu.Inmemory import InMemoryDataset
from paddle_models.MolEncoder import vdWGraph
from paddle_models.PredictionMol import PredModel

import paddle.distributed as dist

    # return retInput, retOutput

def smiles_to_dataset(smiles_list):
    """tbd"""
    data_list = smiles_list
    dataset = InMemoryDataset(data_list=data_list)
    return dataset

def load_json_config(path):
    """tbd"""
    return json.load(open(path, 'r'))

# @paddle.no_grad()
def evaluate(smile_list, pre_model, model_config, collate_fn, k):
    # pre_model.eval()

    t1 = time.time()
    dataset = smiles_to_dataset(smiles_list=smile_list)

    # print('Testing mini_batch', k)
    # print('Transform...')
    transform_fn = PredTransformFn(model_config['pretrain_tasks'])
    dataset.transform(transform_fn, num_workers=1)

    t2 = time.time()

    from tqdm import tqdm

    data_gen = dataset.get_data_loader(
            batch_size = 64, 
            num_workers = 1, 
            shuffle = True, 
            collate_fn = collate_fn)
    
    list_loss = []
    # print('Start testing...')
    pre_model.eval()
    for g, f in data_gen:
        for name in g:
            g[name] = g[name].tensor()
        for name in f:
            f[name] = paddle.to_tensor(f[name], 'float32')
        loss = pre_model(g, f)
        list_loss.append(loss.detach().numpy().mean())
    
    # print('耗时:', round(time.time() - t1, 2))

    return np.mean(list_loss)


def train(smile_list, opt, pre_model, model_config, collate_fn, k):
    
    # pre_model.train()

    from tqdm import tqdm
    t1 = time.time()
    # dataset = smiles_to_dataset(smiles_list=smile_list)

    # print('Training mini_batch', k, '/' , 1080000 // 1000)
    # print('Transform...')
    # transform_fn = GeoPredTransformFn_lulu(model_config['pretrain_tasks'])
    # dataset.transform(transform_fn, num_workers=1)

    f = open('./processed_data/' + str(k) +'.graphdata', 'rb')
    dataset = pkl.load(f)
    f.close()

    t2 = time.time()

    data_gen = dataset.get_data_loader(
            batch_size = 128, 
            num_workers = 1, 
            shuffle = True, 
            collate_fn = collate_fn)

    pre_model.train()
    # list_loss = []
    # print('Start training...')

    for g, f in data_gen:
        for name in g:
            g[name] = g[name].tensor()

        for name in f:
            f[name] = paddle.to_tensor(f[name])

        loss = pre_model(g, f)
        loss.backward()
        opt.step()
        opt.clear_grad()

        # list_loss.append(loss.detach().numpy().mean())
        # print(loss.shape)
    # print('耗时:', round(time.time() - t1, 2))
    # return np.mean(list_loss)

def main():
    lr = 0.001
    test_ratio = 0.01

    from tqdm import tqdm

    smile_list = []
    f = open('./smiles_data.txt')
    lines = f.readlines()
    # print('read smiles...')
    # smile_list_all = [eval(l.strip()) for l in tqdm(lines[:20000])]
    smiles_list_all = lines
    f.close()

    f = open('./2D_poses_cut.txt')
    poses = f.readlines()
    # print('read 2D poses...')
    # poses_all = [eval(p.strip()) for p in tqdm(poses[:20000])]
    poses_all = poses
    f.close()

    f = open('./3D_poses_cut.txt')
    poses_3d = f.readlines()
    # print('read 3D poses...')
    # poses_3d_all = [eval(p.strip()) for p in tqdm(poses_3d[:20000])]
    poses_3d_all = poses_3d
    f.close()

    # print('get smiles raw data...')
    # smile_list = [(smile_list[i][0], smile_list[i][1], poses[i], poses_3d[i]) for i in tqdm(range(len(smile_list)))]

    # test_index = int(len(smile_list) * (1 - test_ratio))
    # train_dataset = smile_list[:test_index]
    # test_dataset = smile_list[test_index:]

    model_config = load_json_config('./config/pretrain.json')
    compound_encoder_config = load_json_config('./config/gnn.json')

    model_config['pretrain_tasks'] = ['Adc', 'Blr', 'van']

    collate_fn = PredCollateFn(
            atom_names=compound_encoder_config['atom_names'],
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            pretrain_tasks=model_config['pretrain_tasks'],
            mask_ratio=model_config['mask_ratio'],
            atom_van_bond_names=compound_encoder_config['atom_van_bond_names'],
            Cm_vocab=model_config['Cm_vocab'])

    # paddle.device.set_device('gpu:0')

    compound_encoder = vdWGraph(compound_encoder_config)
    pre_model = PredModel(model_config, compound_encoder)

    # compound_encoder.set_state_dict(paddle.load('new_save_encoders/no_contras/epoch_22_encoder.pdparams'))
    # pre_model.set_state_dict(paddle.load('new_save_encoders/no_contras/epoch_22_model.pdparams'))

    # opt = paddle.optimizer.Adam(learning_rate=lr, parameters=pre_model.parameters())

    # print('Total param num: %s' % (len(pre_model.parameters())))
    # for i, param in enumerate( pre_model.named_parameters()):
    #     print(i, param[0], param[1].name)
    # print("Train/Test num: %s/%s" % (len(train_dataset), len(test_dataset)))

    # paddle.save(pre_model.state_dict(), './save_pretrain/epoch_-1.pdparams')
    test_epoch_loss = []
    for epoch in range(0, 50):
        t = time.time()
        print('epoch:',epoch)
        test_loss_list = []

        # print('Trainging...')
        size = 1000
        opt = paddle.optimizer.Adam(learning_rate=lr, parameters=pre_model.parameters())
        for i in tqdm(range(1, 1093)):
            # size = 2000

            # print('read smiles and poses...')
            # smiles_list = [eval(l.strip()) for l in smiles_list_all[i:i + size]]
            # # print('read 2D poses...')
            # poses = [eval(l.strip()) for l in poses_all[i:i + size]]
            # # print('read 3D poses...')
            # poses_3d = [eval(l.strip()) for l in poses_3d_all[i:i + size]]
            # # print('get smiles raw data...')
            # smiles_list = [(smiles_list[j][0], smiles_list[j][1], poses[j], poses_3d[j]) for j in range(len(smiles_list))]

            # test_index = int(len(smiles_list) * (1 - test_ratio))
            # train_dataset = smiles_list[:test_index]
            # test_dataset = smiles_list[test_index:]

            train(None, opt, pre_model, model_config, collate_fn, i)

            # batch_test_loss = evaluate(test_dataset, pre_model, model_config, collate_fn, i//size + 1)
            # test_loss_list.append(batch_test_loss)
        
        # print('Testing...')
        # test_loss_list = []
        # for i in range(0, len(test_dataset), 8000):
        #     batch_test_loss = evaluate(test_dataset[i: i+8000], pre_model, model_config, collate_fn, i//8000 + 1)
        #     test_loss_list.append(batch_test_loss)
        
        # test_epoch_loss.append(np.mean(test_loss_list))
        
        # print('epoch:', epoch, '耗时(小时):', round((time.time() - t)/3600, 2), 'testloss:', np.mean(test_loss_list))
        print()

        if len(model_config['pretrain_tasks']) == 4:
            paddle.save(compound_encoder.state_dict(), './new_save_encoders/all/epoch_' + str(epoch) + '_encoder.pdparams')
        elif len(model_config['pretrain_tasks']) == 2:
            paddle.save(compound_encoder.state_dict(), './new_save_encoders/no_contras_van/epoch_' + str(epoch) + '_encoder.pdparams')       
            paddle.save(pre_model.state_dict(), './new_save_encoders/no_contras_van/epoch_' + str(epoch) + '_model.pdparams')        
        elif 'van' not in model_config['pretrain_tasks']:
            paddle.save(compound_encoder.state_dict(), './new_save_encoders/no_van/epoch_' + str(epoch) + '_encoder.pdparams')
        elif 'Adc' not in model_config['pretrain_tasks']:
            paddle.save(compound_encoder.state_dict(), './new_save_encoders/no_adc/epoch_' + str(epoch) + '_encoder.pdparams')
        elif 'Blr' not in model_config['pretrain_tasks']:
            paddle.save(compound_encoder.state_dict(), './new_save_encoders/no_blr/epoch_' + str(epoch) + '_encoder.pdparams')
        elif 'Cm' not in model_config['pretrain_tasks']:
            paddle.save(compound_encoder.state_dict(), './new_save_encoders/no_contras/epoch_' + str(epoch) + '_encoder.pdparams')
            paddle.save(pre_model.state_dict(), './new_save_encoders/no_contras/epoch_' + str(epoch) + '_model.pdparams')
        # else:
        #     paddle.save(compound_encoder.state_dict(), './new_save_encoders/no_contras_van/epoch_' + str(epoch) + '_encoder.pdparams')

    # f = open('test_loss_per_single.txt', 'w')
    # f.write(str(test_epoch_loss))
    # f.close()

if __name__ == '__main__':
    main()