import os
import random

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from parse import args
import torch
from geomloss import SamplesLoss


def sinkhorn_distance(group_embedding, ideal_distribution):
    loss = SamplesLoss("sinkhorn", p=1, blur=0.05)
    return loss(group_embedding, ideal_distribution)


def get_origin_user_interaction_list(graph, user_num):
    user_interactions = {}

    edges = graph.edges()
    src_nodes, dst_nodes = edges

    user_set = set(src_nodes.numpy())  # 7724? 7732

    for user_id in user_set:
        interaction_indices = (src_nodes == user_id).nonzero(as_tuple=True)[0]

        interacted_items = dst_nodes[interaction_indices].numpy()

        user_interactions[user_id] = {str(item_id): 1.0 for item_id in interacted_items}

    np.save("user_interaction_dict.npy", user_interactions)

    return user_interactions, user_set


def get_rec_list(user_set, scores, max_user_id):
    rec_list = {}
    user_list = list(user_set)

    user_indices_tensor = torch.tensor(user_list, dtype=torch.long)

    sorted_indices = torch.argsort(scores[user_indices_tensor], descending=True, dim=1)

    sorted_item_indices = sorted_indices + max_user_id
    # print(sorted_item_indices)

    for idx, user_index in enumerate(user_indices_tensor):
        sorted_user_scores = scores[user_index][sorted_indices[idx]]
        sorted_items = sorted_item_indices[idx]

        formatted_scores = list(zip(map(str, sorted_items.tolist()), sorted_user_scores.tolist()))
        rec_list[int(user_index)] = formatted_scores

    return rec_list


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def mask_test_edges_dgl(graph):
    src, dst = graph.edges()
    edges_all = torch.stack([src, dst], dim=0)
    edges_all = edges_all.t().cpu().numpy()

    all_edge_idx = list(range(edges_all.shape[0]))
    np.random.shuffle(all_edge_idx)

    train_edge_idx = all_edge_idx[:]

    # NOTE: these edge lists only contain single direction of edge!
    return train_edge_idx


def compute_loss_para(adj, device):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm
