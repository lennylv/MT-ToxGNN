from pkgutil import get_data
# from pretraining import load_json_config
import numpy as np
from paddle_models.MolEncoder import vdWGraph
import paddle.nn as nn
import paddle
from paddle_lulu.pahelix_utils import RandomSplitter
from Geo_lulu import DownstreamTransformFn
from Geopredcollatefn import DownstreamCollateFn
from paddle_models.Downstream_LC50DM import DownstreamModel
from scipy.stats import pearsonr
from paddle_lulu.Inmemory import InMemoryDataset
from sklearn.metrics import mean_squared_error
import json

def load_json_config(path):
    """tbd"""
    return json.load(open(path, 'r'))

def get_metric(dataset_name):
    """tbd"""
    # if dataset_name in ['esol', 'freesolv', 'lipophilicity']:
    return 'rmse'

def calc_rmse(labels, preds):
    """tbd"""
    # return np.sqrt(np.mean((preds - labels) ** 2))
    return np.sqrt(mean_squared_error(labels, preds))

def train(
        args, 
        model, label_mean, label_std,
        train_dataset, collate_fn, 
        criterion, encoder_opt, head_opt, task):
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
    model.train()
    loss_return = paddle.to_tensor(0.)
    items = 0
    for atom_bond_graphs, bond_angle_graphs, labels in data_gen:
        if len(labels) < args.batch_size * 0.5:
            continue
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        scaled_labels = labels
        scaled_labels = paddle.to_tensor(scaled_labels, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs, task)
        loss = criterion(preds, scaled_labels)
        items += preds.shape[0]
        loss_return += loss


    return loss_return


def evaluate(
        args, 
        model, label_mean, label_std,
        test_dataset, collate_fn, metric):
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
    # nodes_repr = []
    graph_repr = []
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        scaled_preds = model(atom_bond_graphs, bond_angle_graphs, args.dataset_name)
        # preds = scaled_preds.numpy() * label_std + label_mean
        total_pred.append(scaled_preds.numpy())
        total_label.append(labels.numpy())
        # graph_repr.append(g_repr.numpy())
        # nodes_repr.append(nodes_repr_batch.numpy())
    total_pred = np.concatenate(total_pred, 0).reshape(-1)
    total_label = np.concatenate(total_label, 0).reshape(-1)
    # graph_repr = np.concatenate(graph_repr, 0)
    # np.save('./principle/LD50_' + args.data_type + '_label.npy', total_label)
    # np.save('./principle/' + args.dataset_name + '_' + args.data_type + '_no_fine.npy', graph_repr)
    # if metric == 'rmse':
    # np.save('./dataset/test_nodes_repr.npy', np.concatenate(nodes_repr, 0))
    return calc_rmse(total_label, total_pred), pearsonr(total_label, total_pred)[0]

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

def test_save(args):
    compound_encoder_config = './config/geognn.json'
    compound_encoder_config = load_json_config(compound_encoder_config)

    dataset_name = args.dataset_name

    task_type = 'regr'
    metric = get_metric(dataset_name)

    model_config = './config/mlp.json'
    model_config = load_json_config(model_config)
    model_config['task_type'] = task_type

    label_mean, label_std = get_dataset_stat(dataset_name)

    compound_encoder = vdWGraph(compound_encoder_config)
    if dataset_name == 'LC50DM':
        init_encoder = 'finetune_save/LC50DM_encoder.pdparams'
        init_model = 'finetune_save/LC50DM_model.pdparams'
    elif dataset_name == 'LD50':
        init_encoder = 'finetune_save/LD50_encoder.pdparams'
        init_model = 'finetune_save/LD50_model.pdparams'
    elif dataset_name == 'IGC50':
        init_encoder = 'finetune_save/IGC50_encoder.pdparams'
        init_model = 'finetune_save/IGC50_model.pdparams'
    else:
        init_encoder = 'finetune_save/LC50_encoder.pdparams'
        init_model = 'finetune_save/LC50_model.pdparams'         
    
    compound_encoder.set_state_dict(paddle.load(init_encoder))
    model = DownstreamModel(model_config, compound_encoder)


    model.set_state_dict(paddle.load(init_model))

    transform_fn = DownstreamTransformFn()
    test_dataset = get_dataset(dataset_name, 'testset', args)
    test_dataset.transform(transform_fn, 1)


    # train_dataset = get_dataset(dataset_name, 'trainset', args)
    # train_dataset.transform(transform_fn, 1)

    collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'], 
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            atom_van_bond_names=compound_encoder_config['atom_van_bond_names'],
            task_type=task_type)

    args.data_type = 'test'
    # compound_encoder.set_state_dict(paddle.load('save_encoder/epoch_34.pdparams'))
    test_metric, test_pearson_epoch = evaluate(
                args, model, label_mean, label_std, 
                test_dataset, collate_fn, metric)

    # args.data_type = 'train'
    # compound_encoder.set_state_dict(paddle.load('save_encoder/epoch_34.pdparams'))
    # test_metric, test_pearson_epoch = evaluate(
    #             args, model, label_mean, label_std, 
    #             train_dataset, collate_fn, metric)

    print('pearson:',test_pearson_epoch, 'rmse:',test_metric)
    



def get_dataset(dataset_name, tpe, args):
    # return list: [ (mol, atom_poses, label), ]
    path = './dataset/'  + dataset_name + '/' + tpe + '/'
    poses = np.load(path + 'conformations_2d.npy')
    nodes_number = np.load(path + 'nodes_number.npy')

    # logp = np.load(path + 'logP.npy').reshape([-1,1])
    # molW = np.load(path + 'molW.npy').reshape([-1,1])
    # qed = np.load(path + 'QED.npy').reshape([-1,1])
    # tpsa = np.load(path + 'TPSA.npy').reshape([-1,1])

    import pickle
    f = open(path + 'mols.pkl', 'rb')
    mols = pickle.load(f)
    f.close()

    y = np.load(path + 'y.npy').reshape([-1,1])

    if dataset_name == 'LC50DM' and tpe == 'trainset':
        y = y[:283]

    if dataset_name != args.dataset_name:
        
        path = './dataset/'  + dataset_name + '/testset/'
        poses1 = np.load(path + 'conformations_2d.npy')
        nodes_number1 = np.load(path + 'nodes_number.npy')
        y1 = np.load(path + 'y.npy').reshape([-1,1])
        f = open(path + 'mols.pkl', 'rb')
        mols1 = pickle.load(f)
        f.close()

        poses = np.concatenate([poses, poses1], 0)
        nodes_number = np.concatenate([nodes_number, nodes_number1], 0)
        y = np.concatenate([y, y1],0)
        mols.extend(mols1)

    labels = y
    
    data_list = []
    # if dataset_name == 'LC50DM':
    #     nodes_number = nodes_number[:283]
    for i in range(len(nodes_number)):
        nodes = nodes_number[i]
        pose = poses[i, :nodes, :].tolist()
        label = labels[i]
        mol = mols[i]
        data_list.append((mol, pose, label))
    
    data_list = InMemoryDataset(data_list=data_list)

    # if tpe == 'trainset':
    #     data_list_1 = data_list[:283]
    #     data_list_2 = data_list[283:283+577]
    #     data_list_3 = data_list[283+577:]

    #     return data_list_1, data_list_2, data_list_3
    return data_list

def create_splitter():
    """Return a splitter according to the ``split_type``"""
    # if split_type == 'random':
    splitter = RandomSplitter()
    # elif split_type == 'index':
    #     splitter = IndexSplitter()
    # elif split_type == 'scaffold':
    #     splitter = ScaffoldSplitter()
    # elif split_type == 'random_scaffold':
    #     splitter = RandomScaffoldSplitter()
    # else:
    #     raise ValueError('%s not supported' % split_type)
    return splitter

def get_dataset_stat(dataset_name):
    label = np.load('./dataset/'  + dataset_name + '/trainset/y.npy')
    return np.mean(label), np.std(label)


def clear_main(args):
    encoder_lr = 0.001
    head_lr = 0.001
    compound_encoder_config = './config/geognn.json'
    compound_encoder_config = load_json_config(compound_encoder_config)

    dataset_name = args.dataset_name

    task_type = 'regr'
    metric = get_metric(dataset_name)
        
    # dataset_stat = get_dataset_stat(dataset_name, data_path, task_names)
    # label_mean = np.reshape(dataset_stat['mean'], [1, -1])
    # label_std = np.reshape(dataset_stat['std'], [1, -1])
    model_config = './config/mlp.json'
    model_config = load_json_config(model_config)
    model_config['task_type'] = task_type

    compound_encoder = vdWGraph(compound_encoder_config)
    print('loading init_model..:', args.init_model)
    init_model = './save_encoder/epoch_' + args.init_model + '.pdparams'
    compound_encoder.set_state_dict(paddle.load(init_model))
    model = DownstreamModel(model_config, compound_encoder)

    criterion = nn.MSELoss()

    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)

    encoder_opt = paddle.optimizer.Adam(encoder_lr, parameters=encoder_params)
    head_opt = paddle.optimizer.Adam(head_lr, parameters=head_params)
    print('Total param num: %s' % (len(model.parameters())))
    print('Encoder param num: %s' % (len(encoder_params)))
    print('Head param num: %s' % (len(head_params)))

    # init_model = './save_model/pretrain_models-chemrl_gem/regr.pdparams'
    # compound_encoder.set_state_dict(paddle.load(init_model))

    print('Processing data...')
    train_dataset = get_dataset(dataset_name, 'trainset', args)

    if dataset_name == 'LD50':
        datasets_all = ['LC50DM', 'IGC50', 'LC50', 'LD50']
    else:
        datasets_all = ['LC50DM', 'IGC50', 'LC50',]
    datasets_rest = [l for l in datasets_all if l != args.dataset_name]

    multi_dataset1 = datasets_rest.pop()
    multi_dataset2 = datasets_rest.pop()
    
    train_dataset_668 = get_dataset(multi_dataset1, 'trainset', args)
    train_dataset_wjm = get_dataset(multi_dataset2, 'trainset', args)
    

    transform_fn = DownstreamTransformFn()
    train_dataset.transform(transform_fn, num_workers=args.num_workers)
    train_dataset_668.transform(transform_fn, num_workers=args.num_workers)
    train_dataset_wjm.transform(transform_fn, num_workers=args.num_workers)

    if dataset_name == 'LD50':
        multi_dataset3 = datasets_rest.pop()
        train_dataset_ld = get_dataset(multi_dataset3, 'trainset', args)
        train_dataset_ld.transform(transform_fn, num_workers=args.num_workers)

    test_dataset = get_dataset(dataset_name, 'testset', args)
    test_dataset.transform(transform_fn, 1)

    label_mean, label_std = get_dataset_stat(dataset_name)

    valid_dataset = test_dataset
    # splitter = create_splitter()
    # train_dataset, valid_dataset, test_dataset_none = splitter.split(
    #         train_dataset, frac_train=0.9, frac_valid=0.1, frac_test=0)

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
            task_type=task_type)


    import time
    start_time = time.time()
    # val_pearson = -1
    # patient = 0
    for epoch_id in range(args.max_epoch):

        train_loss = train(
                args, model, label_mean, label_std, 
                train_dataset, collate_fn, 
                criterion, encoder_opt, head_opt, args.dataset_name)

        train_loss += train(
                args, model, label_mean, label_std, 
                train_dataset_668, collate_fn, 
                criterion, encoder_opt, head_opt, multi_dataset1)
        
        train_loss += train(
                args, model, label_mean, label_std, 
                train_dataset_wjm, collate_fn, 
                criterion, encoder_opt, head_opt, multi_dataset2)

        if dataset_name == 'LD50':
            train_loss += train(
                    args, model, label_mean, label_std, 
                    train_dataset_ld, collate_fn, 
                    criterion, encoder_opt, head_opt, multi_dataset3)

        train_loss.backward()
        encoder_opt.step()
        head_opt.step()
        encoder_opt.clear_grad()
        head_opt.clear_grad()
  

        val_metric, val_pearson_epoch = evaluate(
                args, model, label_mean, label_std, 
                valid_dataset, collate_fn, metric)
        
        # test_metric, test_pearson_epoch = evaluate(
        #         args, model, label_mean, label_std, 
        #         test_dataset, collate_fn, metric)

        test_rmse, test_pearson = val_metric, val_pearson_epoch

        list_val_metric.append(val_metric)
        end_time = time.time()
        print()
        print("epoch:%s train/loss:%s" % (epoch_id, train_loss.numpy()) ,'time:',round((end_time - start_time), 2))
        print("         val/%s:%s pearson:%s" % ( metric, val_metric, val_pearson_epoch))

        start_time = end_time

    # if test_pearson > 0.823:
    temp = int(round(test_pearson, 3) * 1000)
    paddle.save(compound_encoder.state_dict(), './finetune_save/' + args.dataset_name + '_' + str(temp) + '_encoder.pdparams')
    paddle.save(model.state_dict(), './finetune_save/' + args.dataset_name + '_' + str(temp) + '_model.pdparams')

    f = open('record.txt', 'a')
    f.write('dataset: ' + args.dataset_name + ' init_model:' + str(args.init_model)+' seed:' + str(args.seed) +' rmse:' + str(round(test_rmse, 2)) + ' personr:' + str(round(test_pearson,3)) + '\n')
    f.close()

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['train', 'data'], default='train')

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=800)
    parser.add_argument("--dataset_name", default='LC50')
    parser.add_argument("--init_model", default='34')
    parser.add_argument("--seed", type=int, default=312134)

    import random
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    # you can choose epochs from 500 to 800, there is no significant different results for these epochs
    # Following AGBT-FP and other methods, we do not split validation set in Four toxicity-related datasets and LogP
    # and we save the model after the final epoch

    # however, you can split validation set from the training set in line 331 - 334 of the code
    if args.re_train == 1:
        clear_main(args)
    else:
        test_save(args)

if __name__ == '__main__':
    main()
