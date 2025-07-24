import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors

def read_PipeFile(debug):
    # debug 情况下直接读 txt文件
    if debug:
        # 打开并读取txt文件
        with open('恒洋热电百步线.txt', 'r', encoding='utf-8') as file:
            # 使用json.load()方法将JSON字符串解析为Python对象
            data = json.load(file)

        # 读取节点和边的信息
        nodelist = data['nodelist']
        edgelist = data['linklist']
        # 把两者转化成df好用
        nodeDF = pd.DataFrame(nodelist)
        edgeDF = pd.DataFrame(edgelist)
        # 得到节点和边的ID的Series
        # 前面的'id' + 和后面的 .replace('-', '_') 为了换成跟pipe计算结果一样的 id
        # 也就是 '04c4e973-1451-4130-a2b2-7a0ce394c87f' 变成 id04c4e973_1451_4130_a2b2_7a0ce394c87f
        nodeID = 'id' + nodeDF['id'].str.replace('-', '_')       # Series, 95节点
        # edgeID 应该用不着, 因为 edge 是通过 起始点的 node_id 来描述的
        edgeID = 'id' + edgeDF['id'].str.replace('-', '_')       # Series, 94边
        edgeDF['sourceid'] = 'id' + edgeDF['sourceid'].str.replace('-', '_')
        edgeDF['targetid'] = 'id' + edgeDF['targetid'].str.replace('-', '_')

        # 得到坐标信息
        nodeParas = [json.loads(nodeDF['parameter'][i]) for i in range(len(nodeDF))]
        nodePSTs = []
        for nodePara in nodeParas:
            nodePSTs.append([nodePara['styles']['position']['x'], nodePara['styles']['position']['y']])
        nodePSTs = pd.Series(nodePSTs)

        return nodeDF, edgeDF, nodeID, edgeID, nodePSTs
    # 在线运行时再说
    else:
        pass

def make_edge_attr(edgeDF, edgeST_idx):
    # 建立连接关系矩阵, [2, num_edges]
    edge_attr = pd.DataFrame(edgeST_idx.T, columns=['from', 'to'])

    # 读取边的长度和 id 并添加到 edge_attr
    edge_attr['cost'] = [json.loads(edge)['parameter']['Length'] for edge in edgeDF['parameter'].values]
    edge_attr['edge_id'] = [json.loads(edge)['id'] for edge in edgeDF['parameter'].values]
    edge_attr['edge_id'] = 'id' + edge_attr['edge_id'].str.replace('-', '_')

    # 转换成csv格式写入 data 文件夹下的 PipeH25 文件夹, 命名为 PipeH25, 如果没有这个路径就新建, 如果在线运行用后面的路径
    """os.makedirs(dir_path, exist_ok=True)
    edge_attr.to_csv(dir_path + '/PipeH25.csv', index=False)"""


    return edge_attr


# 通过聚类划分用户不同聚集地
def main(debug):

    # 读节点和边数据 df
    nodeDF, edgeDF, nodeID, edgeID, nodePSTs = read_PipeFile(debug)

    # 其实可以直接用 Series 去检索，但是我觉得用字典方便
    # 创建一个字典，将节点ID映射为索引
    nodeID_to_index = {val: idx for idx, val in enumerate(nodeID)}
    # 创建一个字典，将边ID映射为索引, 跟 edgeID 一样, 可能也用不到
    edgeID_to_index = {val: idx for idx, val in enumerate(edgeID)}

    # 边起始点的矩阵，端点是ID
    edgeST_val = np.array([edgeDF['sourceid'].values, edgeDF['targetid'].values])
    # 边起始点的矩阵，端点是索引
    edgeST_idx = np.array([[nodeID_to_index[val] for val in subarray] for subarray in edgeST_val])

    # 得到边的起始点，距离和id
    edge_attr = make_edge_attr(edgeDF, edgeST_idx)
    # 定义聚类函数
    def apply_clustering_and_visualize(algorithm_name, clustering, title):
        """应用聚类算法并可视化结果"""
        # 执行聚类
        clusters = clustering.fit_predict(distance_matrix)
        cluster_dict = {node: cluster for node, cluster in zip(G.nodes(), clusters)}
        
        # 可视化
        plt.figure(figsize=(10, 8))
        
        # 绘制管网拓扑结构
        pos = node_positions
        edge_labels = {(u, v): d['length'] for u, v, d in G.edges(data=True)}
        
        # 绘制节点，按聚类结果着色
        unique_clusters = np.unique(clusters)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            nodes_in_cluster = [node for node, c in cluster_dict.items() if c == cluster]
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=nodes_in_cluster,
                node_color=[colors[i]],
                node_size=500,
                label=f'聚类 {cluster}'
            )
        
        # 绘制边和标签
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='SimHei')
        
        plt.title(title, fontsize=14)
        plt.legend(scatterpoints=1, loc='upper right')
        plt.grid(alpha=0.3)
        plt.axis('off')
        plt.tight_layout()
        
        return cluster_dict


    # 基于管网最短路径法
    # 构建网络
    G = nx.Graph()
    G.add_nodes_from(nodeID.index)
    edges4G = [(edge_attr['from'].iloc[i], edge_attr['to'].iloc[i], {'length':edge_attr['cost'].iloc[i]}) for i in range(len(edgeID))]
    G.add_edges_from(edges4G)
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))
    n = len(G.nodes())
    distance_matrix = np.zeros((n, n))
    for i, node_i in enumerate(G.nodes()):
        for j, node_j in enumerate(G.nodes()):
            distance_matrix[i, j] = path_lengths[node_i][node_j]
    
    # 基于欧氏距离
    # 获取节点坐标
    node_positions = nodePSTs.to_dict()
    """X = np.array([node_positions[node] for node in G.nodes()])
    distance_matrix = euclidean_distances(X)"""

    # 设置中文字体支持
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

    # 计算每个点到第k近邻的距离
    """k = 4  # 通常取min_samples-1
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(distance_matrix)
    distances, indices = nn.kneighbors()
    distances = np.sort(distances[:, k-1], axis=0)

    # 绘制K-dist图
    plt.plot(distances)
    plt.xlabel('点的索引')
    plt.ylabel(f'到第{k}近邻的距离')
    plt.grid(True)
    plt.show()"""

    # 树状结构
    from scipy.cluster.hierarchy import dendrogram, linkage

    # 基于距离矩阵生成链接矩阵
    Z = linkage(distance_matrix, method='average')

    # 绘制树状图
    """plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=list(G.nodes()))
    plt.title('层次聚类树状图')
    plt.xlabel('节点')
    plt.ylabel('距离')
    plt.axhline(y=150, color='r', linestyle='--', label='聚类阈值')
    plt.legend()
    plt.show()"""
    # 从树状图中选择合适的截断高度确定簇数

    # 1. DBSCAN聚类（基于密度）
    dbscan_dict = apply_clustering_and_visualize(
        "DBSCAN",
        DBSCAN(eps=185, min_samples=3, metric='precomputed'),
        "DBSCAN聚类结果（eps=150, min_samples=2）"
    )

    # 2. 层次聚类（凝聚式）
    # 感觉这个更好，了解一下原理
    hierarchical_dict = apply_clustering_and_visualize(
        "层次聚类",
        AgglomerativeClustering(
            n_clusters=10,
            compute_full_tree=True,
            metric="precomputed",  # 替换affinity为metric
            linkage='average'
        ),
        "层次聚类结果（k=17, 平均连接）"
    )

    # 3. K-means聚类（基于质心）
    from sklearn.cluster import KMeans

    # 使用节点位置作为特征（实际应用中可替换为距离矩阵）
    X = np.array([node_positions[node] for node in G.nodes()])

    kmeans_dict = apply_clustering_and_visualize(
        "K-means",
        KMeans(n_clusters=15, random_state=42),
        "K-means聚类结果（k=15）"
    )

    plt.show()

    # 打印聚类结果
    #print("\nDBSCAN聚类结果:", dbscan_dict)
    #print("层次聚类结果:", hierarchical_dict)
    #print("K-means聚类结果:", kmeans_dict)

    # 仅保留出口的聚类（层次聚类）
    # 找出edge_attr中存在于耗端但是又不在产端的
    # 1. 合并所有节点ID
    all_nodes = pd.concat([edge_attr['from'], edge_attr['to']]).unique()
    # 2. 计算每个节点的出度（在from列中出现的次数）
    out_degree = edge_attr['from'].value_counts().reindex(all_nodes, fill_value=0)
    # 3. 筛选出度为0的节点（仅作为消耗端的节点）
    sinks = edge_attr[out_degree == 0]['from'].values
    print(sinks)
    # 4. 筛选出hierarchical_dict中的耗端
    filtered_dict = {key: hierarchical_dict[key] for key in sinks if key in hierarchical_dict}
    classes = set(list(filtered_dict.values()))

    print("筛选结果:", classes)

    pass



debug = 0

if __name__ == '__main__':
    debug = 1
    main(debug)