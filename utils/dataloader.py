import dgl
import scipy.sparse as sp
import torch
from parse import args
import numpy as np
from torch.utils.data import Dataset
from random import shuffle, choice


def load_datasets(data_name):
    dataset_paths = {
        "yelp2018": {
            "train": "./dataset/yelp2018/yelp_train_data.bin",
            "test": "./dataset/yelp2018/yelp_test_data.bin",
        },
        "douban": {
            "train": "./dataset/douban/douban_train_data.bin",
            "test": "./dataset/douban/douban_test_data.bin",
        },
        "kuairec": {
            "train": "./dataset/kuairec/kuairec_train_data.bin",
            "test": "./dataset/kuairec/kuairec_test_data.bin",
        },
        "food": {
            "train": "./dataset/food/food_train_data.bin",
            "test": "./dataset/food/food_test_data.bin",
        },
        # Please add your own dataset path in there, and you need first add you dataset into ./dataset.
    }

    if data_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {data_name}")

    selected_paths = dataset_paths[data_name]
    datasets = {key: dgl.load_graphs(path)[0][0] for key, path in selected_paths.items()}
    return datasets


def prepare_data(graph, device):
    num_nodes = graph.number_of_nodes()
    feats = graph.ndata['feat'].to(device)
    edge_index = torch.stack(graph.edges()).to(device)
    return feats, edge_index, num_nodes


def prepare_loss_parameters(graph, device):
    # 获取图中的所有边
    src, dst = graph.edges()
    edges_all = torch.stack([src, dst], dim=0)
    edges_all = edges_all.t().cpu().numpy()

    # 随机打乱边的顺序
    all_edge_idx = list(range(edges_all.shape[0]))
    np.random.shuffle(all_edge_idx)

    # 使用所有边作为训练边
    train_edge_idx = all_edge_idx[:]

    # 创建包含选定边的子图，并获取其邻接矩阵
    train_graph = dgl.edge_subgraph(graph, train_edge_idx, relabel_nodes=False).to(device)
    adj = train_graph.adjacency_matrix().to_dense().to(device)

    # 计算正负样本的权重和归一化系数
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight

    return weight_tensor, norm, adj


class DRO_dataloader(Dataset):
    def __init__(self, train_interaction_matrix, num_user, num_item):
        """
        初始化数据集
        :param interaction_matrix: scipy.sparse.csr_matrix，用户-物品交互矩阵
        """
        self.interaction_matrix = train_interaction_matrix
        self.num_users, self.num_items = num_user, num_item
        self.user_indices = []
        self.pos_item_indices = []
        self.neg_item_indices = []
        self.pair_stage = np.load("./dataset/{0}/interaction_group_dict.npy".format(args.dataset),
                                  allow_pickle=True).item()

        # 预处理数据，生成正负样本
        # for user_id in range(self.num_users):
        #     # 使用稀疏矩阵格式获取正样本索引
        #     pos_items = interaction_matrix[user_id].indices
        #     remaining_items = list(set(range(self.num_items)) - set(pos_items))
        #     if not remaining_items:
        #         continue  # 如果没有剩余的物品可供选择，则跳过
        #     neg_items = np.random.choice(remaining_items, size=len(pos_items), replace=False)
        #     self.user_indices.extend([user_id] * len(pos_items))
        #     self.pos_item_indices.extend(pos_items)
        #     self.neg_item_indices.extend(neg_items)
        for user_id in range(self.num_users):
            # 使用稀疏矩阵格式获取正样本索引
            pos_items = self.interaction_matrix[user_id].indices

            # 确保所有物品 ID 在 0 到 num_item-1 范围内
            all_items = set(range(self.num_items))

            # 获取负样本候选集
            remaining_items = list(all_items - set(pos_items))

            if not remaining_items:
                continue  # 如果没有剩余的物品可供选择，则跳过

            # 随机选择负样本，确保其范围在有效物品 ID 范围内
            neg_items = np.random.choice(remaining_items, size=len(pos_items), replace=False)

            # 更新用户、正样本和负样本的索引列表
            self.user_indices.extend([user_id] * len(pos_items))
            self.pos_item_indices.extend(pos_items)
            self.neg_item_indices.extend(neg_items)

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        user_id = self.user_indices[idx]
        pos_item_id = self.pos_item_indices[idx]
        neg_item_id = self.neg_item_indices[idx]

        return user_id, pos_item_id, neg_item_id, self.pair_stage[
            (user_id, pos_item_id)]


def get_user_item_matrix(graph):
    edge_index = torch.stack(graph.edges())

    max_user = max(set(edge_index[0].tolist()))
    max_item = max(set(edge_index[1].tolist()))

    num_user = len(set(edge_index[0].tolist()))
    num_item = len(set(edge_index[1].tolist()))
    print("the number of users: {0}".format(num_user))
    print("the number of items: {0}".format(num_item))

    row, col, entries = [], [], []
    for edge in edge_index.t().tolist():
        row.append(edge[0])
        col.append(edge[1] - max_user - 1)
        entries.append(1.0)
    row_np = np.array(row)
    col_np = np.array(col)

    temp_interaction_mat = sp.csr_matrix((entries, (row_np, col_np+num_user)),
                                         shape=(num_user + num_item, num_user + num_item),
                                         dtype=np.float32)

    # save_sparse_matrix_to_pickle(interaction_mat, 'interaction_matrix.pkl')
    # print("最大用户编号{0}".format(max_user))
    # print("最大物品编号{0}".format(find_max_number(col)))

    interaction_mat = temp_interaction_mat + temp_interaction_mat.T

    return interaction_mat, num_user, num_item


def get_train_user_item_matrix(graph):
    edge_index = torch.stack(graph.edges())

    max_user = max(set(edge_index[0].tolist()))
    max_item = max(set(edge_index[1].tolist()))

    num_user = len(set(edge_index[0].tolist()))
    num_item = len(set(edge_index[1].tolist()))

    row, col, entries = [], [], []
    interaction_list = []  # 新增一个列表来保存交互信息

    for edge in edge_index.t().tolist():
        user_id = edge[0]
        item_id = edge[1] - max_user - 1
        rating = 1.0  # 假设所有交互的评分为 1.0，可以根据实际情况调整
        row.append(user_id)
        col.append(item_id)
        entries.append(rating)

        # 保存交互信息为 [user_id, item_id, rating] 的格式
        # interaction_list.append([user_id, item_id, rating])

    row_np = np.array(row)
    col_np = np.array(col)

    interaction_mat = sp.csr_matrix((entries, (row_np, col_np)),
                                    shape=(num_user , num_item),
                                    dtype=np.float32)

    return interaction_mat  # 返回稀疏矩阵和交互列表


def normalize_graph_mat(adj_mat):
    shape = adj_mat.shape
    rowsum = np.array(adj_mat.sum(1))
    if shape[0] == shape[1]:
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
    else:
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
    return norm_adj_mat


def ideal_distribution_cal(embeddings, dataset):
    # 加载中介中心性数据
    centrality = np.load('./dataset/{0}/centrality_dict_softmax.npy'.format(args.dataset), allow_pickle=True)

    centrality = centrality.item()
    # 计算前 10% 的节点数量
    num_top_nodes = int(len(centrality) * 0.05)

    # 获取前 10% 节点，按中心性值排序
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:num_top_nodes]

    # 提取这些节点对应的嵌入
    ideal_distribution = embeddings[top_nodes]

    return ideal_distribution


# def ideal_distribution_cal(dataset):
#     # 加载中介中心性数据
#     centrality = np.load('../dataset/yelp2018/centrality_dict_softmax.npy', allow_pickle=True)
#
#     print(centrality)
#
#     centrality = centrality.item()
#     # 计算前 10% 的节点数量
#     num_top_nodes = int(len(centrality) * 0.1)
#
#     # 获取前 10% 节点，按中心性值排序
#     top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:num_top_nodes]
#
#     # 提取这些节点对应的嵌入
#     # ideal_distribution = embeddings[top_nodes]
#
#     return 0
#
#
# if __name__ == '__main__':
#     ideal_distribution_cal('yelp2018')


def generate_interaction_matrix_from_dgl(graph, user_num, item_num):
    # src_nodes, dst_nodes = graph.edges()
    # src_nodes = src_nodes.cpu().numpy()
    # dst_nodes = dst_nodes.cpu().numpy()
    src, dst = graph.edges()
    src_nodes = src.cpu().numpy()
    dst = dst.cpu().numpy()

    ratings = np.ones_like(src_nodes, dtype=np.float32)

    temp_interaction_mat = sp.csr_matrix(
        (ratings, (src_nodes, dst)),
        shape=(user_num + item_num, item_num + user_num),
        dtype=np.float32
    )
    interaction_matrix = temp_interaction_mat + temp_interaction_mat.T

    return interaction_matrix
