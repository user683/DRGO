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


def group_users_by_kmeans(user_embeddings, interaction_df, n_clusters=5, random_state=0):
    # 训练 KMeans 模型
    user_embeddings = user_embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(user_embeddings)

    # 获取每个用户的聚类标签
    user_clusters = kmeans.labels_

    # 将稀疏矩阵转换为 COO 格式（坐标格式）
    interaction_coo = interaction_df.tocoo()

    # 创建 DataFrame，包含 user_id, item_id 和对应的值
    interaction_df = pd.DataFrame({
        'user_id': interaction_coo.row,  # 行索引对应 user_id
        'item_id': interaction_coo.col,  # 列索引对应 item_id
        'interaction_value': interaction_coo.data  # 矩阵中的值
    })

    # 只保留 user_id 和 item_id 两列
    interaction_df = interaction_df[['user_id', 'item_id']]

    # 将用户ID与对应的分组标签（Cluster ID）关联
    user_group_mapping = pd.DataFrame({
        'user_id': interaction_df['user_id'].unique(),
        'cluster_id': user_clusters
    })

    # 将分组信息映射到用户-物品交互数据
    interaction_df = pd.merge(interaction_df, user_group_mapping, on='user_id')

    # 将用户-物品交互数据格式化为 {(user_id, item_id): user_group_id} 的形式
    interaction_group_dict = {
        (row['user_id'], row['item_id']): row['cluster_id']
        for _, row in interaction_df.iterrows()
    }
    # 返回字典 将字典保存
    np.save('./dataset/{0}/interaction_group_dict.npy'.format(args.dataset), interaction_group_dict,
            allow_pickle=True)

    print("聚类完成")
    return interaction_group_dict


def get_origin_user_interaction_list(graph, user_num):
    user_interactions = {}

    edges = graph.edges()
    src_nodes, dst_nodes = edges

    user_set = set(src_nodes.numpy()) # 7724? 7732

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
