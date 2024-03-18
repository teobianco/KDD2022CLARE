import argparse
from datetime import datetime
import random
import numpy as np
import torch
from Rewriter import CommRewriting
from Locator import CommMatching
from utils import split_communities, eval_scores, prepare_data
import os
from utils.helper_funcs import assign_free_gpus

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write2file(comms, filename):
    with open(filename, 'w') as fh:
        content = '\n'.join([', '.join([str(i) for i in com]) for com in comms])
        fh.write(content)


def read4file(filename):
    with open(filename, "r") as file:
        pred = [[int(node) for node in x.split(', ')] for x in file.read().strip().split('\n')]
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General Config
    parser.add_argument("--seed", type=int, help="seed", default=0)
    parser.add_argument("--device", dest="device", type=str, help="training device", default="cuda:0")
    parser.add_argument("--dataset", type=str, help="dataset", default="amazon")
    #   --in CLARE paper, we predict 1000 communities from 100 communities as a default setting
    parser.add_argument("--num_pred", type=int, help="pred size", default=1000)
    parser.add_argument("--num_train", type=int, help="pred size", default=90)
    parser.add_argument("--num_val", type=int, help="pred size", default=10)
    parser.add_argument("--already_train_test", type=bool, help="If the train and test communities are already defined", default=True)
    parser.add_argument("--multiplier", type=float, help="multiplier", default=1.0)

    # Community Locator related
    #   --GNNEncoder Setting
    parser.add_argument("--gnn_type", type=str, help="type of convolution", default="GCN")
    parser.add_argument("--n_layers", type=int, help="number of gnn layers", default=2)
    parser.add_argument("--hidden_dim", type=int, help="training hidden size", default=64)
    parser.add_argument("--output_dim", type=int, help="training hidden size", default=64)
    #   --Order Embedding Setting
    parser.add_argument("--margin", type=float, help="margin loss", default=0.6)
    #   --Generation
    parser.add_argument("--comm_max_size", type=int, help="Community max size", default=12)
    #   --Training
    parser.add_argument("--locator_lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--locator_epoch", type=int, default=30)
    parser.add_argument("--locator_batch_size", type=int, default=256)

    # Community Rewriter related
    parser.add_argument("--agent_lr", type=float, help="CommR learning rate", default=1e-3)
    #    -- for DBLP, the setting of n_eisode and n_epoch is a little picky
    parser.add_argument("--n_episode", type=int, help="number of episode", default=10)
    parser.add_argument("--n_epoch", type=int, help="number of epoch", default=1000)
    parser.add_argument("--gamma", type=float, help="CommR gamma", default=0.99)
    parser.add_argument("--max_step", type=int, help="", default=10)
    parser.add_argument("--max_rewrite_step", type=int, help="", default=4)
    parser.add_argument("--commr_path", type=str, help="CommR path", default="")

    # Save log
    parser.add_argument("--writer_dir", type=str, help="Summary writer directory", default="")

    args = parser.parse_args()
    seed_all(args.seed)

    # If run on GPU, we need to assign free GPUs
    if args.device == "cuda:0":
        assign_free_gpus(max_gpus=1)

    print('= ' * 20)
    print('##  Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print(args)

    if not os.path.exists(f"ckpts/{args.dataset}"):
        os.mkdir(f"ckpts/{args.dataset}")

    f_list = []
    j_list = []
    nmi_list = []

    time_len = len(os.listdir(f"./dataset/{args.dataset}/"))
    for time in range(time_len):
        print(f'## Start Timestep {time} ...')
        # args.writer_dir = f"ckpts/{args.dataset}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        args.writer_dir = f"ckpts/{args.dataset}/{time}"
        os.mkdir(args.writer_dir)
        args.comm_max_size = 20 if args.dataset.startswith("lj") else 12

        ##########################################################
        ################### Step 1 Load Data #####################
        ##########################################################
        num_node, num_edge, num_community, graph_data, nx_graph, communities, mapping = prepare_data(args.dataset, time)
        print(f"Finish loading data: {graph_data}\n")
        train_comms, val_comms, test_comms = split_communities(communities, args.num_train, args.num_val, args.already_train_test, args.dataset, time, mapping)
        # DEVO SCRIVERE QUELLA NUOVA PENSANDO A COME ESTRARRE VAL_COMMS
        print(f"Split dataset: #Train {len(train_comms)}, #Val {len(val_comms)}, #Test {len(test_comms)}\n")

        ##########################################################
        ################### Step 2 Train Locator##################
        ##########################################################
        CommM_obj = CommMatching(args, graph_data, train_comms, val_comms, device=torch.device(args.device))
        CommM_obj.train()
        pred_comms = CommM_obj.predict_community(nx_graph, args.comm_max_size)
        f1, jaccard, onmi = eval_scores(pred_comms, test_comms, train_comms, val_comms, tmp_print=True)
        metrics_string = '_'.join([f'{x:0.4f}' for x in [f1, jaccard, onmi]])
        write2file(pred_comms, args.writer_dir + "/CommM_" + metrics_string + '.txt')

        ##########################################################
        ################### Step 3 Train Rewriter#################
        ##########################################################
        cost_choice = "f1"  # or you can change to "jaccard"
        feat_mat = CommM_obj.generate_all_node_emb().detach().cpu().numpy()  # all nodes' embedding
        CommR_obj = CommRewriting(args, nx_graph, feat_mat, train_comms, val_comms, pred_comms, cost_choice)
        CommR_obj.train()
        rewrite_comms = CommR_obj.get_rewrite()
        # DA AGGIUNGERE CHE FARE CON LE RIPETIZIONI DEI NODI IN COMUNITà DIVERSE  --> LASCIARE COSì
        # CAPIRE SE ELIMINARE NODI CHE SONO NEL TRAIN SET  --> FATTO, PROVARE SE FUNZIONA
        f1, jaccard, onmi = eval_scores(rewrite_comms, test_comms, train_comms, val_comms, tmp_print=True)
        f_list.append(f1)
        j_list.append(jaccard)
        nmi_list.append(onmi)
        metrics_string = '_'.join([f'{x:0.4f}' for x in [f1, jaccard, onmi]])
        write2file(rewrite_comms, args.writer_dir + f"/CommR_{cost_choice}_" + metrics_string + '.txt')

    print(f'Mean F1: {np.mean(f_list)}')
    print(f'Std F1: {np.std(f_list)}')
    print(f'Mean Jaccard: {np.mean(j_list)}')
    print(f'Std Jaccard: {np.std(j_list)}')
    print(f'Mean ONMI: {np.mean(nmi_list)}')
    print(f'Std ONMI: {np.std(nmi_list)}')
    print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)
