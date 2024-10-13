import torch
import torch.nn as nn
import numpy as np

#
# class LGCN_Encoder(nn.Module):
#     def __init__(self, emb_size, n_layers, num_user, num_item):
#         super(LGCN_Encoder, self).__init__()
#         self.user_num = num_user
#         self.item_num = num_item
#         self.latent_size = emb_size
#         self.layers = n_layers
#         self.embedding_dict = self._init_model()
#
#     def _init_model(self):
#         initializer = nn.init.xavier_uniform_
#         embedding_dict = nn.ParameterDict({
#             'user_emb': nn.Parameter(initializer(torch.empty(self.user_num, self.latent_size))),
#             'item_emb': nn.Parameter(initializer(torch.empty(self.item_num, self.latent_size))),
#         })
#         return embedding_dict
#
#     @staticmethod
#     def convert_sparse_mat_to_tensor(sparse_mat):
#         sparse_mat = sparse_mat.tocoo()
#         indices = torch.from_numpy(np.vstack((sparse_mat.row, sparse_mat.col)).astype(np.int64))
#         values = torch.from_numpy(sparse_mat.data)
#         shape = torch.Size(sparse_mat.shape)
#         return torch.sparse_coo_tensor(indices, values, shape)
#
#     def forward(self, normal_adj):
#         sparse_norm_adj = self.convert_sparse_mat_to_tensor(normal_adj).to(
#             torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#         ego_embeddings = torch.cat(
#             [self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
#         all_embeddings = [ego_embeddings]
#         print(ego_embeddings.shape)
#         print(sparse_norm_adj.shape)
#
#         for k in range(self.layers):
#             ego_embeddings = torch.sparse.mm(sparse_norm_adj, ego_embeddings)
#             all_embeddings += [ego_embeddings]
#         all_embeddings = torch.stack(all_embeddings, dim=1)
#         all_embeddings = torch.mean(all_embeddings, dim=1)
#         return all_embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
#
class LGCN_Encoder(nn.Module):
    def __init__(self, emb_size, n_negs, n_layers, num_user, num_item):
        super(LGCN_Encoder, self).__init__()
        self.emb_size = emb_size
        self.layers = n_layers
        self.n_negs = n_negs
        self.user_num = num_user
        self.item_num = num_item
        self.embedding_dict = self._init_model()
        self.dropout = nn.Dropout(0.1)
        # self.sparse_norm_adj = self.convert_sparse_mat_to_tensor(normal_adj).to(
        #     torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self,normal_adj):
        sparse_norm_adj = self.convert_sparse_mat_to_tensor(normal_adj).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        user_embs = [self.embedding_dict['user_emb']]
        item_embs = [self.embedding_dict['item_emb']]
        # adj = self._sparse_dropout(self.sparse_norm_adj, 0.5)
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(sparse_norm_adj, ego_embeddings)
            ego_embeddings = self.dropout(ego_embeddings)
            user_embs.append(ego_embeddings[:self.user_num])
            item_embs.append(ego_embeddings[self.user_num:])
        user_embs = torch.stack(user_embs, dim=1)
        user_embs = torch.mean(user_embs, dim=1)
        return user_embs, item_embs

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(device)
        return out * (1. / (1 - rate))

    def negative_mixup(self, user, pos_item, neg_item, user_emb, item_emb):
        user_list = user.detach().tolist()
        pos_item_list = pos_item.detach().tolist()
        neg_item_list = neg_item.detach().tolist()
        u_emb = user_emb[user_list]
        negs = []
        for k in range(self.layers + 1):
            pos_emb = item_emb[k][pos_item_list]
            neg_emb = item_emb[k][neg_item_list]
            neg_emb = neg_emb.reshape(-1, self.n_negs, self.emb_size)
            alpha = torch.rand_like(neg_emb).to(device)
            neg_emb = alpha * pos_emb.unsqueeze(dim=1) + (1 - alpha) * neg_emb
            scores = (u_emb.unsqueeze(dim=1) * neg_emb).sum(dim=-1)
            indices = torch.max(scores, dim=1)[1].detach()
            chosen_neg_emb = neg_emb[torch.arange(neg_emb.size(0)), indices]
            negs.append(chosen_neg_emb)
        item_emb = torch.stack(item_emb, dim=1)
        item_emb = torch.mean(item_emb, dim=1)
        negs = torch.stack(negs, dim=1)
        negs = torch.mean(negs, dim=1)
        return u_emb, item_emb[pos_item], negs

    def get_embeddings(self, normal_adj):
        sparse_norm_adj = self.convert_sparse_mat_to_tensor(normal_adj).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        return all_embeddings

    @staticmethod
    def convert_sparse_mat_to_tensor(sparse_mat):
        sparse_mat = sparse_mat.tocoo()
        indices = torch.from_numpy(np.vstack((sparse_mat.row, sparse_mat.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mat.data)
        shape = torch.Size(sparse_mat.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

