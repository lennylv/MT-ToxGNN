# vdWGraph

--------------------------------------------------------------------------------------
Pre-train data availability:

smiles with 2D and 3D conformers: https://pan.baidu.com/s/1avy6ek6Dl1Sn7uuzYSYw6w
password: wdnm

Pre-train the model:
```
python pretraining.py
```
To save training time, you can generate all needed data for all smiles first:
```
python prepare_graph_data.py
```
Or download our processes data: https://pan.baidu.com/s/1JyEitczI3ih5vbBDMMaQig password:wdnm

cat the files
```
cat logs.tar.bz2.* > new_tar.tar
tar xvf new_tar.tar -C ./processed_data/
```
--------------------------------------------------------------------------------------

Reproduce results:

four toxicity-datasets and logP:

The finetune_**.py are used to quickly reproduce the results:
```
python finetune_Toxicity.py --dataset LC50
```
you can re-train the model by:
```
python finetune_Toxicity.py --dataset LC50 --re_train 1
```

FreeSolv, Lipop and BBBP:

```
python finetune_FreeSol.py --random False
python finetune_Lip.py --scaffold True
python finetune_BBBP.py
```

Some details:

1. Lip data is saved as data/lip.pkl, you can unzip data/lip.zip to use it to save time
2. The processed BBBP data is also provided in data/*

3. The trainset of logp is > 25 mb, and you can unzip dataset/logp/trainset.zip to use it

4. The trained model are provided in Downstream/* 

5. The pre-trained model are provided in save_encoder/*, we provide 3 pre-trained models.

--------------------------------------------------------------------------------------

Dependency:
```
pip install paddlepaddle
pip install rdkit-pypi
pip install pgl
pip install deepchem
```

--------------------------------------------------------------------------------------
Any more questions, please let me know:
20205227080@stu.suda.edu.cn
