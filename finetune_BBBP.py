from pkgutil import get_data
# from pretraining import load_json_config
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_score, recall_score, roc_auc_score, log_loss

from paddle_models.MolEncoder import vdWGraph
import paddle.nn as nn
import paddle
from paddle_lulu.pahelix_utils import RandomSplitter, ScaffoldSplitter
from Geo_lulu import DownstreamTransformFn
from Geopredcollatefn import DownstreamCollateFn
from paddle_models.Downstream_BBBP import DownstreamModel
from scipy.stats import pearsonr
from paddle_lulu.Inmemory import InMemoryDataset

import paddle.static as static
import random

def load_json_config(path):
    """tbd"""
    import json
    return json.load(open(path, 'r'))

def test_save(args):
    compound_encoder_config = './config/geognn.json'
    compound_encoder_config = load_json_config(compound_encoder_config)

    dataset_name = args.dataset_name

    metric = get_metric(dataset_name)
    # task_names = get_downstream_task_names(dataset_name, data_path) 

    model_config = './config/mlp.json'
    model_config = load_json_config(model_config)

    compound_encoder = vdWGraph(compound_encoder_config)
    init_model = './finetune_save/BBBP_encoder.pdparams'
    compound_encoder.set_state_dict(paddle.load(init_model))
    model = DownstreamModel(model_config, compound_encoder)

    model.set_state_dict(paddle.load('./finetune_save/BBBP_model.pdparams'))

    transform_fn = DownstreamTransformFn()
    _, _, test_dataset = get_dataset(dataset_name)
    test_dataset.transform(transform_fn, 1)

    collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'], 
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            atom_van_bond_names=compound_encoder_config['atom_van_bond_names'],
            task_type='class')

    acc, roc = evaluate(args, model, test_dataset, collate_fn, metric)
    
    print('acc:', acc, 'roc:', roc)

def get_metric(dataset_name):
    """tbd"""
    # if dataset_name in ['esol', 'freesolv', 'lipophilicity']:
    return 'cross'

def calc_accruacy(labels, preds):

    auc_roc = roc_auc_score(labels, preds)
    preds[preds<0.5] = 0
    preds[preds>0.5] = 1

    acc = accuracy_score(labels, preds)
    # auc_roc = roc_auc_score(labels, preds)
    return acc, auc_roc

def train( args, model, train_dataset, collate_fn, 
        criterion, encoder_opt, head_opt):
    """
    Define the train function 
    Args:
        args,model,train_dataset,collate_fn,criterion,encoder_opt,head_opt;
    Returns:
        the average of the list loss
    """
    data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=collate_fn)
    list_loss = []
    nodes_repr = []
    model.train()
    loss_return = paddle.to_tensor(0.)
    items = 0
    for atom_bond_graphs, bond_angle_graphs, labels in data_gen:
        # if len(labels) < args.batch_size * 0.5:
        #     continue
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        # scaled_labels = (labels - label_mean) / (label_std + 1e-5)
        scaled_labels = labels
        scaled_labels = paddle.to_tensor(scaled_labels, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs)
        # print(preds.shape, scaled_labels.shape)
        loss = criterion(preds, scaled_labels)
        items += preds.shape[0]
        # loss.backward()
        # encoder_opt.step()
        # head_opt.step()
        # encoder_opt.clear_grad()
        # head_opt.clear_grad()
        list_loss.append(loss.numpy())

        loss_return += loss
        # list_loss.append(loss)
        # nodes_repr.append(nodes_repr_batch.numpy())
    # np.save('./dataset/train_nodes_repr.npy', np.concatenate(nodes_repr, 0))
    # return np.mean(list_loss)
    # print(len(list_loss))

    return loss_return


def evaluate( args, model, test_dataset, collate_fn, metric):
    """
    Define the evaluate function
    In the dataset, a proportion of labels are blank. So we use a `valid` tensor 
    to help eliminate these blank labels in both training and evaluation phase.
    """
    data_gen = test_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False,
            collate_fn=collate_fn)
    total_pred = []
    total_label = []
    nodes_repr = []
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        scaled_preds = model(atom_bond_graphs, bond_angle_graphs)
        total_pred.append(scaled_preds.numpy())
        total_label.append(labels.numpy())
        # nodes_repr.append(nodes_repr_batch.numpy())
    total_pred = np.concatenate(total_pred, 0).reshape(-1)
    total_label = np.concatenate(total_label, 0).reshape(-1)
    # if metric == 'rmse':
    # np.save('./dataset/test_nodes_repr.npy', np.concatenate(nodes_repr, 0))
    return calc_accruacy(total_label, total_pred)

def exempt_parameters(src_list, ref_list):
    """Remove element from src_list that is in ref_list"""
    res = []
    for x in src_list:
        flag = True
        for y in ref_list:
            if x is y:
                flag = False
                break
        if flag:
            res.append(x)
    return res

def get_dataset(dataset_name):

    path = './dataset/'  + dataset_name + '/'
    poses = np.load(path + 'conformations_2d.npy')
    nodes_number = np.load(path + 'nodes_number.npy')

    train = np.load(path + 'train.npy')
    valid = np.load(path + 'valid.npy')
    test = np.load(path + 'test.npy')
    import pickle
    f = open(path + 'mols.pkl', 'rb')
    mols = pickle.load(f)
    f.close()

    y = np.load(path + 'y.npy').reshape([-1,1])
    labels = y
    
    data_list = []

    for i in range(len(nodes_number)):
        nodes = nodes_number[i]
        pose = poses[i, :nodes, :].tolist()
        label = labels[i]
        mol = mols[i]
        data_list.append((mol, pose, label))
    
    data_list = InMemoryDataset(data_list=data_list)

    train_dataset = data_list[train.tolist()]
    valid_dataset = data_list[valid.tolist()]
    test_dataset = data_list[test.tolist()]

    return train_dataset, valid_dataset, test_dataset

def create_splitter():
    """Return a splitter according to the ``split_type``"""
    # splitter = RandomSplitter()
    splitter = ScaffoldSplitter()
    return splitter

def get_dataset_stat(dataset_name):
    label = np.load('./dataset/'  + dataset_name + '/y.npy')
    return np.mean(label), np.std(label)


def clear_main(args):
    encoder_lr = 0.001
    head_lr = 0.001
    compound_encoder_config = './config/geognn.json'
    compound_encoder_config = load_json_config(compound_encoder_config)

    dataset_name = args.dataset_name

    metric = get_metric(dataset_name)
    model_config = './config/mlp.json'
    model_config = load_json_config(model_config)

    compound_encoder = vdWGraph(compound_encoder_config)
    print('loading init_model..:', args.init_model)
    init_model = './save_encoder/epoch_' + args.init_model + '.pdparams'
    compound_encoder.set_state_dict(paddle.load(init_model))
    model = DownstreamModel(model_config, compound_encoder)

    criterion = nn.BCELoss()

    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)

    encoder_opt = paddle.optimizer.Adam(encoder_lr, parameters=encoder_params)
    head_opt = paddle.optimizer.Adam(head_lr, parameters=head_params)
    print('Total param num: %s' % (len(model.parameters())))
    print('Encoder param num: %s' % (len(encoder_params)))
    print('Head param num: %s' % (len(head_params)))

    print('Processing data...')
    train_dataset, valid_dataset, test_dataset = get_dataset(dataset_name)

    transform_fn = DownstreamTransformFn()
    train_dataset.transform(transform_fn, num_workers=args.num_workers)
    valid_dataset.transform(transform_fn, num_workers=args.num_workers)
    test_dataset.transform(transform_fn, num_workers=args.num_workers)

    # label_mean, label_std = get_dataset_stat(dataset_name)

    # valid_dataset = test_dataset
    splitter = create_splitter()
    train_dataset, valid_dataset, test_dataset = splitter.split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    print("Train/Valid/Test num: %s/%s/%s" % (
            len(train_dataset), len(valid_dataset), len(test_dataset)))

    ### start train
    list_val_metric = []
    collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'], 
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            atom_van_bond_names=compound_encoder_config['atom_van_bond_names'],
            task_type='class')
    
    import time
    start_time = time.time()
    valid_acc = 0.
    valid_roc = 0.
    patient = 0
    for epoch_id in range(args.max_epoch):

        train_loss = train(
                args, model, train_dataset, collate_fn, 
                criterion, encoder_opt, head_opt)
        
        train_loss.backward()

        encoder_opt.step()
        head_opt.step()
        encoder_opt.clear_grad()
        head_opt.clear_grad()

        valid_acc_epoch, valid_roc_epoch = evaluate(
                args, model, valid_dataset, collate_fn, metric)
        
        test_acc_epoch, test_roc_epoch = evaluate(
                args, model, test_dataset, collate_fn, metric)

        end_time = time.time()
        print()
        print("epoch:%s train/loss:%s" % (epoch_id, train_loss.numpy()[0]) ,'time:',round((end_time - start_time), 2))
        print("         val/acc:%s auc_roc:%s" % ( valid_acc_epoch, valid_roc_epoch))

        if valid_roc_epoch > valid_roc:
            patient = 0
            valid_acc = valid_acc_epoch
            valid_roc = valid_roc_epoch
            test_acc = test_acc_epoch
            test_roc = test_roc_epoch
            print("         test/acc:%s auc_roc:%s" % (test_acc, test_roc))
            paddle.save(compound_encoder.state_dict(), './finetune_save/' + str(round(test_roc,3))[2:] + '_BBBP_encoder.pdparams')
            paddle.save(model.state_dict(), './finetune_save/' + str(round(test_roc,3))[2:] + '_BBBP_model.pdparams')
        else:
            patient += 1
            if patient == 150:break
        # print("epoch:%s test/%s:%s" % (epoch_id, metric, test_metric))
        # print("epoch:%s test/%s_by_eval:%s" % (epoch_id, metric, test_metric_by_eval))

        start_time = end_time

    # if test_pearson > 0.804:
    #     paddle.save(compound_encoder.state_dict(), './finetune_save/' + str(round(test_pearson,3))[2:] + '_FDA_encoder.pdparams')
    #     paddle.save(model.state_dict(), './finetune_save/' + str(round(test_pearson,3))[2:] + '_FDA_model.pdparams')

    f = open('record.txt', 'a')
    f.write('dataset: ' + args.dataset_name + ' init_model:' + str(args.init_model)+' seed:' + str(args.seed) +' acc:' + str(round(test_acc, 3)) + ' auc_roc:' + str(round(test_roc,3)) + '\n')
    f.close()

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['train', 'data'], default='train')

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--re_train", type=int, default=0)
    parser.add_argument("--dataset_name", default='BBBP')
    parser.add_argument("--init_model", default='38')
    parser.add_argument("--seed", type=int, default=798328)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    # clear_main(args)
    
    if args.re_train == 1:
        clear_main(args)
    else:
        test_save(args)
if __name__ == '__main__':
    main()
