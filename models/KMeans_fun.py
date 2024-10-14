import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import time
from parse import args


def generate_interaction_group_dict(interaction_mat, n_clusters=5, n_components=64):
    """
    Perform dimensionality reduction and clustering on user-item interaction data, and generate a dictionary of
    user-item group information.
    :param interaction_mat: User-item interaction matrix, of type csr_matrix
    :param n_clusters: Number of clusters for KMeans clustering, default is 10
    :param n_components: Dimensionality after reduction, default is 32
    :param save_path: Path to save the grouping dictionary, default is None, which means it will not be saved
    :return: Generated user-item interaction group information dictionary {(user_id, item_id): user_group_id}"
    """

    # start_time = time.time()

    # Use TruncatedSVD for dimensionality reduction
    svd = TruncatedSVD(n_components=n_components)
    user_embeddings = svd.fit_transform(interaction_mat)

    # Use KMeans to cluster user embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    user_clusters = kmeans.fit_predict(user_embeddings)

    # Associate user IDs with the corresponding group labels (Cluster IDs)
    user_ids = np.arange(interaction_mat.shape[0])
    user_group_mapping = pd.DataFrame({
        'user_id': user_ids,
        'cluster_id': user_clusters
    })

    # Format the user-item interaction information as a dictionary in the format {(user_id, item_id): user_group_id}
    interaction_df = pd.DataFrame(interaction_mat.tocoo().row, columns=['user_id'])
    interaction_df['item_id'] = interaction_mat.tocoo().col
    interaction_df = pd.merge(interaction_df, user_group_mapping, on='user_id')

    interaction_group_dict = {
        (row['user_id'], row['item_id']): row['cluster_id']
        for _, row in interaction_df.iterrows()
    }

    # If a save path is specified, the dictionary will be saved as a .npy file.
    np.save("./dataset/{0}/interaction_group_dict.npy".format(args.dataset), interaction_group_dict, allow_pickle=True)

    # Print clustering time
    # end_time = time.time()
    # print(f"Clustering time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")

    return interaction_group_dict






#     return interaction_group_dict
