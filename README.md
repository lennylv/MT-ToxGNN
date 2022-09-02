# vdWGraph

--------------------------------------------------------------------------------------
Pre-train data availability:

smiles with 2D and 3D conformers: https://pan.baidu.com/s/1avy6ek6Dl1Sn7uuzYSYw6w
password: wdnm

Pre-train the model:

- python pretraining.py

To save training time, you can generate all needed data for all smiles first:

- python prepare_graph_data.py

--------------------------------------------------------------------------------------

Reproduce results:

The finetune_**.py are used to quickly reproduce the results:

- python finetune_Toxicity.py --dataset LC50

you can re-train the model by:

- python finetune_Toxicity.py --dataset LC50 --retrain 1

--------------------------------------------------------------------------------------

Dependency:

- pip install paddlepaddle
- pip install rdkit-pypi
- pip install pgl
