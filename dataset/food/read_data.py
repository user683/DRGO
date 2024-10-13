import dgl
import  numpy as np
import torch
import networkx as nx
from joblib import Parallel, delayed

# 加载数据集
dataset_paths = {"train": "food_train_data.bin",
                 "test": "food_test_data.bin", }

datasets = {key: dgl.load_graphs(path)[0][0] for key, path in dataset_paths.items()}

train_graph = datasets['train']


# 提取边索引
edge_index = torch.stack(train_graph.edges())
print(edge_index)

np.save("edge_index.npy", edge_index)


import numpy as np
import networkx as nx
import time

def format_time(seconds):
    """将秒数格式化为 时:分:秒"""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def softmax(x):
    """对输入的数组 x 应用 Softmax 函数"""
    e_x = np.exp(x - np.max(x))  # 为了数值稳定性，减去最大值
    return e_x / e_x.sum()

# 记录开始时间
print("开始计算图的中介中心性")
start_time = time.time()

# 加载边索引数据
edge_index = np.load("edge_index.npy")

# 构建图
G = nx.Graph()
edges = list(zip(edge_index[0], edge_index[1]))
G.add_edges_from(edges)

# 计算中介中心性
centrality = nx.betweenness_centrality(G, k=100, normalized=True)

# 对中介中心性结果进行 Softmax 归一化
centrality_values = np.array(list(centrality.values()))
centrality_softmax = softmax(centrality_values)
#
# # 将归一化后的结果保存为字典，格式为 {user_id: softmax_value}
centrality_dict_softmax = {node: value for node, value in zip(centrality.keys(), centrality_softmax)}

# 保存结果为 .npy 文件
np.save("centrality_dict_softmax.npy", centrality_dict_softmax )

# 记录结束时间
end_time = time.time()

print(f"中介中心性计算完成并进行了 Softmax 归一化，耗时: {format_time(end_time - start_time)}")

