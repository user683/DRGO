import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import time
from parse import args


def generate_interaction_group_dict(interaction_mat, n_clusters=5, n_components=64):
    """
    对用户-物品交互数据进行降维和聚类，并生成用户-物品分组信息字典。

    :param interaction_mat: 用户-物品交互矩阵，类型为 csr_matrix
    :param n_clusters: KMeans 聚类的簇数，默认为 10
    :param n_components: 降维后的维度，默认为 32
    :param save_path: 保存分组字典的路径，默认为 None，不保存
    :return: 生成的用户-物品交互分组信息字典 {(user_id, item_id): user_group_id}
    """

    # start_time = time.time()

    # 使用 TruncatedSVD 进行降维
    svd = TruncatedSVD(n_components=n_components)
    user_embeddings = svd.fit_transform(interaction_mat)

    # 使用 KMeans 对用户嵌入进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    user_clusters = kmeans.fit_predict(user_embeddings)

    # 将用户ID与对应的分组标签（Cluster ID）关联
    user_ids = np.arange(interaction_mat.shape[0])
    user_group_mapping = pd.DataFrame({
        'user_id': user_ids,
        'cluster_id': user_clusters
    })

    # 将用户-物品交互信息格式化为 {(user_id, item_id): user_group_id} 的字典
    interaction_df = pd.DataFrame(interaction_mat.tocoo().row, columns=['user_id'])
    interaction_df['item_id'] = interaction_mat.tocoo().col
    interaction_df = pd.merge(interaction_df, user_group_mapping, on='user_id')

    interaction_group_dict = {
        (row['user_id'], row['item_id']): row['cluster_id']
        for _, row in interaction_df.iterrows()
    }

    # 如果指定了保存路径，则将字典保存为 .npy 文件

    np.save("./dataset/{0}/interaction_group_dict.npy".format(args.dataset), interaction_group_dict, allow_pickle=True)

    # 打印聚类耗时
    # end_time = time.time()
    # print(f"聚类耗时: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")

    return interaction_group_dict

# def generate_interaction_group_dict(interaction_mat, user_embeddings, n_clusters=5, n_components=64):
#     """
#     对用户-物品交互数据进行降维和聚类，并生成用户-物品分组信息字典。
#
#     :param interaction_mat: 用户-物品交互矩阵，类型为 csr_matrix
#     :param n_clusters: KMeans 聚类的簇数，默认为 10
#     :param n_components: 降维后的维度，默认为 32
#     :param save_path: 保存分组字典的路径，默认为 None，不保存
#     :return: 生成的用户-物品交互分组信息字典 {(user_id, item_id): user_group_id}
#     """
#
#     # start_time = time.time()
#
#     # # 使用 TruncatedSVD 进行降维
#     # svd = TruncatedSVD(n_components=n_components)
#     # user_embeddings = svd.fit_transform(interaction_mat)
#
#     # 使用 KMeans 对用户嵌入进行聚类
#     kmeans = KMeans(n_clusters=n_clusters, random_state=64)
#     user_clusters = kmeans.fit_predict(user_embeddings.cpu().numpy())
#
#     # 将用户ID与对应的分组标签（Cluster ID）关联
#     user_ids = np.arange(interaction_mat.shape[0])
#     user_group_mapping = pd.DataFrame({
#         'user_id': user_ids,
#         'cluster_id': user_clusters
#     })
#
#     # 将用户-物品交互信息格式化为 {(user_id, item_id): user_group_id} 的字典
#     interaction_df = pd.DataFrame(interaction_mat.tocoo().row, columns=['user_id'])
#     interaction_df['item_id'] = interaction_mat.tocoo().col
#     interaction_df = pd.merge(interaction_df, user_group_mapping, on='user_id')
#
#     interaction_group_dict = {
#         (row['user_id'], row['item_id']): row['cluster_id']
#         for _, row in interaction_df.iterrows()
#     }
#
#     # 如果指定了保存路径，则将字典保存为 .npy 文件
#
#     np.save("./dataset/{0}/interaction_group_dict.npy".format(args.dataset), interaction_group_dict, allow_pickle=True)
#
#     # 打印聚类耗时
#     # end_time = time.time()
#     # print(f"聚类耗时: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
#
#     return interaction_group_dict
