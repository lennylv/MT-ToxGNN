# vdWGraph

--------------------------------------------------------------------------------------
Pre-train data availability:

smiles with 2D and 3D conformers: https://pan.baidu.com/s/1avy6ek6Dl1Sn7uuzYSYw6w
password: wdnm

--------------------------------------------------------------------------------------

Reproduce results:

The finetune_**.py are used to quickly reproduce the results:

- python finetune_Toxicity.py --dataset LC50

you can re-train the model by:

- python finetune_Toxicity.py --dataset LC50 --retrain 1

--------------------------------------------------------------------------------------

Pre-train the model:

- python pre-train.py
