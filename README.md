# DynCLARE 

This repository contains the source code of DynCLARE method. It is an extended version of CLARE method presented in paper [CLARE: A Semi-supervised Community Detection Algorithm](https://dl.acm.org/doi/10.1145/3534678.3539370_). This source code is strongly based on CLARE method source code from repository https://github.com/FDUDSDE/KDD2022CLARE.

CLARE is an algorithm performing semi-supervised community detection and we extended it in two main ways:
- we made it able to work with **dynamic graphs** both in a naive and in a more structured way
- we added the possibility to exploit embeddings generated with other dynamic graph learning algorithms inside the *Community Locator*


## Table of Contents

- [KDD2022CLARE](#KDD2022CLARE)
  - [Table of Contents](#table-of-contents)
  - [Main Innovations](#paper-intro)
  - [Run CLARE](#run-clare) 
    - [Environmental Requirement](#environmental-requirement)
    - [Run the code](#run-the-code)
    - [Sample log](#sample-log)


 ## Main Innovations

As already said there are 2 main innovations of DynCLARE w.r.t. CLARE: **Dynamic Graphs** applicability and possibility to **use other embedding methods**


### Dynamic Graphs

Differently from CLARE, which is static, DynCLARE does semi-supervised community detection on **dynamic graphs**. It can perform this task in two ways:
- by simply performing CLARE algorithm in each timestep separately (**naive** way)
- by performing CLARE algorithm in each timestep separately, but inizializing RL learnable weigths to those of previous timestep, thus preserving memory of previous past community structures (**structured** way)


### Other Embedding Methods

Original CLARE method exploits **order embedding** to provide community vector reperesentations to the *Community Locator*. Despite its good result, we think that adding the possibility to use embeddings generated with other methods can expand the fields in which this algorithm has high-quality performances

Moreover, in the context of this master's thesis, this feature allows to use embedding generated with other methods that are already able to manage dynamic graphs. Therefore, this let the temporal dimension of dynamic graphs get inside the algorithm not only in the *Community Rewriter* part (inside the RL, as described in previous section), but also inside the *Community Locator*. In this way we make DynCLARE an algorithm much more oriented towards semi-supervised dynamic community detection.


## Run CLARE


This repository contains the following contents:

```
.
├── Locator                       --> (The folder containing Community Locator source code)
├── Rewriter                      --> (The folder containing Community Rewriter source code)
├── ckpts                         --> (The folder saving checkpoint files)
├── dataset                       --> (The folder containing used datasets)
├── main.py                       --> (The main code file. The code is run through this file)
├── dataset_ONMI.py               --> (The file containing code to evaluate datasets communities stability through ONMI)
└── utils                         --> (The folder containing utils functions)


```
You have to create a `ckpts` folder to save contents.

For our experimental datasets, raw datasets are available at SNAP(http://snap.stanford.edu/data/index.html) and pre-processing details are explained in our paper.
We select LiveJournal, DBLP and Amazon, in the **Networks with ground-truth communities** part.
We provide 7 datasets. Each of them contains a community file `{name}-1.90.cmty.txt` and an edge file `{name}-1.90.ungraph.txt`.
If you want to run on your **own datasets**, you have to convert your own data into our format, *i.e.*, **a community file** where each line contains a unique community and **an edge file** where each line contains an edge.


### Environmental Requirement

0. You need to set up the environment for running the experiments (Python 3.7 or above)

1. Install **Pytorch** with version 1.8.0 or later

2.  Install **torch-geometric** package with version 2.0.1

    Note that it may need to appropriately install the package `torch-geometric` based on the CUDA version (or CPU version if GPU is not available). Please refer to the official website https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for more information of installing prerequisites.


### Run the code

Execute the `main.py` file

```
python main.py --dataset=amazon  
```

Main arguments already in CLARE (for more argument options, please refer to `main.py`):

```
--dataset [amazon, dblp, lj, amazon_dblp, dblp_amazon, dblp_lj, lj_dblp]: the dataset to run
--num_pred / num_train / num_val: the numbers for prediction, training, and validation
--locator_epoch: number of epochs to train Community Locator (default setting 30)
--n_layers: ego-net dimensions & number of GNN layers (default 2)
--agent_lr: the learning rate of Community Rewriter
--max_step: the maximum operations (EXPAND/EXCLUDE) of rewriting a community
```

Arguments we added in DynCLARE

```
--already_train_test
--multiplier
--memory
--method
```



  
