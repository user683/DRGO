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
        # Please add your own dataset path in there, and you need first add your dataset into ./dataset.
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
    # Get all the edges in the graph
    src, dst = graph.edges()
    edges_all = torch.stack([src, dst], dim=0)
    edges_all = edges_all.t().cpu().numpy()

    # Randomly shuffle the order of edges
    all_edge_idx = list(range(edges_all.shape[0]))
    np.random.shuffle(all_edge_idx)

    # Use all edges as training edges
    train_edge_idx = all_edge_idx[:]

    # Create a subgraph containing the selected edges and get its adjacency matrix
    train_graph = dgl.edge_subgraph(graph, train_edge_idx, relabel_nodes=False).to(device)
    adj = train_graph.adjacency_matrix().to_dense().to(device)

    # Calculate weights and normalization coefficient for positive and negative samples
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight

    return weight_tensor, norm, adj


class DRO_dataloader(Dataset):
    def __init__(self, train_interaction_matrix, num_user, num_item):
        """
        Initialize the dataset
        :param interaction_matrix: scipy.sparse.csr_matrix, user-item interaction matrix
        """
        self.interaction_matrix = train_interaction_matrix
        self.num_users, self.num_items = num_user, num_item
        self.user_indices = []
        self.pos_item_indices = []
        self.neg_item_indices = []
        self.pair_stage = np.load("./dataset/{0}/interaction_group_dict.npy".format(args.dataset),
                                  allow_pickle=True).item()

        # Preprocess data to generate positive and negative samples
        for user_id in range(self.num_users):
            # Use sparse matrix format to get positive sample indices
            pos_items = self.interaction_matrix[user_id].indices

            # Ensure all item IDs are within the range of 0 to num_item-1
            all_items = set(range(self.num_items))

            # Get negative sample candidates
            remaining_items = list(all_items - set(pos_items))

            if not remaining_items:
                continue  # Skip if no remaining items are available

            # Randomly select negative samples, ensuring they are within valid item ID range
            neg_items = np.random.choice(remaining_items, size=len(pos_items), replace=False)

            # Update index lists for users, positive samples, and negative samples
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
    print("The number of users: {0}".format(num_user))
    print("The number of items: {0}".format(num_item))

    row, col, entries = [], [], []
    for edge in edge_index.t().tolist():
        row.append(edge[0])
        col.append(edge[1] - max_user - 1)
        entries.append(1.0)
    row_np = np.array(row)
    col_np = np.array(col)

    temp_interaction_mat = sp.csr_matrix((entries, (row_np, col_np + num_user)),
                                         shape=(num_user + num_item, num_user + num_item),
                                         dtype=np.float32)

    interaction_mat = temp_interaction_mat + temp_interaction_mat.T

    return interaction_mat, num_user, num_item


def get_train_user_item_matrix(graph):
    edge_index = torch.stack(graph.edges())

    max_user = max(set(edge_index[0].tolist()))
    max_item = max(set(edge_index[1].tolist()))

    num_user = len(set(edge_index[0].tolist()))
    num_item = len(set(edge_index[1].tolist()))

    row, col, entries = [], [], []
    interaction_list = []  # Add a list to store interaction information

    for edge in edge_index.t().tolist():
        user_id = edge[0]
        item_id = edge[1] - max_user - 1
        rating = 1.0  # Assume the rating for all interactions is 1.0, can be adjusted based on actual situation
        row.append(user_id)
        col.append(item_id)
        entries.append(rating)

        # Save interaction information in the format [user_id, item_id, rating]

    row_np = np.array(row)
    col_np = np.array(col)

    interaction_mat = sp.csr_matrix((entries, (row_np, col_np)),
                                    shape=(num_user, num_item),
                                    dtype=np.float32)

    return interaction_mat  # Return sparse matrix and interaction list


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
    # Load centrality data
    centrality = np.load('./dataset/{0}/centrality_dict_softmax.npy'.format(args.dataset), allow_pickle=True)

    centrality = centrality.item()
    # Calculate the number of top 10% nodes
    num_top_nodes = int(len(centrality) * 0.05)

    # Get top 10% nodes, sorted by centrality value
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:num_top_nodes]

    # Extract embeddings corresponding to these nodes
    ideal_distribution = embeddings[top_nodes]

    return ideal_distribution


def generate_interaction_matrix_from_dgl(graph, user_num, item_num):
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
