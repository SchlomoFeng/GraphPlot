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
        # 修改文件名为新的更新文件
        with open('/home/runner/work/GraphPlot/GraphPlot/0708烟台S4_updated.json', 'r', encoding='utf-8') as file:
            # 使用json.load()方法将JSON字符串解析为Python对象
            data = json.load(file)

        # 读取节点和边的信息
        nodelist = data['nodelist']
        edgelist = data['linklist']
        # 把两者转化成df好用
        nodeDF = pd.DataFrame(nodelist)
        edgeDF = pd.DataFrame(edgelist)
        
        # 直接使用原始ID，不做转换
        nodeID = nodeDF['id']  # Series，直接使用原始UUID
        edgeID = edgeDF['id']  # Series，直接使用原始UUID
        
        # 边的起始点也使用原始ID
        edgeDF['sourceid_original'] = edgeDF['sourceid']
        edgeDF['targetid_original'] = edgeDF['targetid']

        # 得到坐标信息
        nodeParas = []
        nodePSTs = []
        
        for i in range(len(nodeDF)):
            try:
                if isinstance(nodeDF['parameter'].iloc[i], str):
                    para = json.loads(nodeDF['parameter'].iloc[i])
                else:
                    para = nodeDF['parameter'].iloc[i]
                nodeParas.append(para)
                
                # 提取位置信息
                if 'styles' in para and 'position' in para['styles']:
                    position = para['styles']['position']
                    nodePSTs.append([position['x'], position['y']])
                else:
                    # 如果没有位置信息，使用默认值
                    nodePSTs.append([0, 0])
                    print(f"警告: 节点 {nodeDF['id'].iloc[i]} 没有位置信息，使用默认坐标 (0, 0)")
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"解析节点 {nodeDF['id'].iloc[i]} 的参数时出错: {e}")
                nodeParas.append({})
                nodePSTs.append([0, 0])
        
        nodePSTs = pd.Series(nodePSTs)
        
        # Extract node types and filter for Stream nodes only
        nodeTypes = []
        for para in nodeParas:
            node_type = para.get('type', 'Unknown')
            nodeTypes.append(node_type)

        # Filter for Stream nodes only
        stream_mask = [node_type == 'Stream' for node_type in nodeTypes]
        original_node_count = len(nodeDF)
        stream_node_count = sum(stream_mask)
        
        print(f"原始节点数: {original_node_count}")
        print(f"Stream节点数: {stream_node_count}")
        print(f"过滤掉的节点数: {original_node_count - stream_node_count}")
        
        # Apply filter to all node-related data structures
        nodeDF = nodeDF[stream_mask].reset_index(drop=True)
        nodePSTs = nodePSTs[stream_mask].reset_index(drop=True)
        
        # Update nodeID to only include Stream nodes
        nodeID = nodeDF['id']

        return nodeDF, edgeDF, nodeID, edgeID, nodePSTs
    # 在线运行时再说
    else:
        pass

def make_edge_attr(edgeDF, nodeID_to_index):
    # 建立连接关系矩阵，使用索引
    edge_data = []
    skipped_edges = 0
    
    for i in range(len(edgeDF)):
        source_id = edgeDF['sourceid_original'].iloc[i]
        target_id = edgeDF['targetid_original'].iloc[i]
        
        # 检查节点是否存在 (只有Stream节点会在nodeID_to_index中)
        if source_id in nodeID_to_index and target_id in nodeID_to_index:
            from_idx = nodeID_to_index[source_id]
            to_idx = nodeID_to_index[target_id]
            
            # 解析边的参数获取长度
            try:
                if isinstance(edgeDF['parameter'].iloc[i], str):
                    edge_para = json.loads(edgeDF['parameter'].iloc[i])
                else:
                    edge_para = edgeDF['parameter'].iloc[i]
                
                # 尝试获取长度信息
                length = edge_para.get('parameter', {}).get('Length', 1.0)
                if length is None or length <= 0:
                    length = 1.0  # 默认长度
                    
                edge_data.append({
                    'from': from_idx,
                    'to': to_idx,
                    'cost': float(length),
                    'edge_id': edgeDF['id'].iloc[i]
                })
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"解析边 {edgeDF['id'].iloc[i]} 的参数时出错: {e}，使用默认长度1.0")
                edge_data.append({
                    'from': from_idx,
                    'to': to_idx,
                    'cost': 1.0,
                    'edge_id': edgeDF['id'].iloc[i]
                })
        else:
            skipped_edges += 1
    
    print(f"跳过了 {skipped_edges} 条边（连接非Stream节点）")
    
    edge_attr = pd.DataFrame(edge_data)
    return edge_attr

# 通过聚类划分用户不同聚集地
def main(debug):

    # 读节点和边数据 df
    nodeDF, edgeDF, nodeID, edgeID, nodePSTs = read_PipeFile(debug)
    
    print(f"读取到 {len(nodeDF)} 个节点和 {len(edgeDF)} 条边")

    # 创建一个字典，将节点ID映射为索引
    nodeID_to_index = {val: idx for idx, val in enumerate(nodeID)}
    index_to_nodeID = {idx: val for idx, val in enumerate(nodeID)}

    # 得到边的起始点，距离和id
    edge_attr = make_edge_attr(edgeDF, nodeID_to_index)
    
    if len(edge_attr) == 0:
        print("错误: 没有有效的边数据")
        return
    
    print(f"处理了 {len(edge_attr)} 条有效边")

    # 定义聚类函数
    def apply_clustering_and_visualize(algorithm_name, clustering, title):
        """应用聚类算法并可视化结果"""
        # 执行聚类
        if algorithm_name == "K-means":
            # K-means使用坐标数据
            X = np.array([nodePSTs.iloc[i] for i in range(len(nodePSTs))])
            clusters = clustering.fit_predict(X)
        else:
            # 其他算法使用距离矩阵
            clusters = clustering.fit_predict(distance_matrix)
        
        cluster_dict = {index_to_nodeID[i]: cluster for i, cluster in enumerate(clusters)}
        
        # 可视化
        plt.figure(figsize=(12, 10))
        
        # 准备节点位置字典（索引->坐标）
        pos = {i: nodePSTs.iloc[i] for i in range(len(nodePSTs))}
        
        # 绘制节点，按聚类结果着色
        unique_clusters = np.unique(clusters)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            nodes_in_cluster = [idx for idx, c in enumerate(clusters) if c == cluster]
            if nodes_in_cluster:  # 确保聚类不为空
                cluster_positions = [pos[node] for node in nodes_in_cluster]
                x_coords = [p[0] for p in cluster_positions]
                y_coords = [p[1] for p in cluster_positions]
                # 使用圆形标记('o')绘制Stream节点
                plt.scatter(x_coords, y_coords, 
                           c=[colors[i]], s=50, marker='o',
                           label=f'聚类 {cluster}', alpha=0.7)
        
        # 绘制边
        for _, edge in edge_attr.iterrows():
            from_pos = pos[edge['from']]
            to_pos = pos[edge['to']]
            plt.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], 
                    'k-', alpha=0.3, linewidth=0.5)
        
        plt.title(title, fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        
        # 保存图片而不是显示
        filename = f"/tmp/{title.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"保存图片: {filename}")
        plt.close()  # 关闭图片以节省内存
        
        return cluster_dict

    # 基于管网最短路径法
    # 构建网络
    G = nx.Graph()
    
    # 添加节点（使用索引）
    G.add_nodes_from(range(len(nodeID)))
    
    # 添加边
    edges4G = [(edge_attr['from'].iloc[i], edge_attr['to'].iloc[i], {'length': edge_attr['cost'].iloc[i]}) 
               for i in range(len(edge_attr))]
    G.add_edges_from(edges4G)
    
    print(f"图构建完成: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
    
    # 检查图的连通性
    if not nx.is_connected(G):
        print("警告: 图不连通，将使用最大连通分量")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"最大连通分量: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
    
    # 计算最短路径
    try:
        path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))
        n = len(G.nodes())
        distance_matrix = np.zeros((n, n))
        
        # 构建节点列表（按图中的顺序）
        nodes_list = list(G.nodes())
        
        for i, node_i in enumerate(nodes_list):
            for j, node_j in enumerate(nodes_list):
                if node_j in path_lengths[node_i]:
                    distance_matrix[i, j] = path_lengths[node_i][node_j]
                else:
                    # 如果节点不连通，使用一个大值
                    distance_matrix[i, j] = np.inf
        
        print("距离矩阵计算完成")
        
    except Exception as e:
        print(f"计算最短路径时出错: {e}")
        print("使用欧氏距离作为备选方案")
        
        # 备选方案：使用欧氏距离
        X = np.array([nodePSTs.iloc[i] for i in G.nodes()])
        distance_matrix = euclidean_distances(X)

    # 设置中文字体支持
    plt.rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']

    try:
        # 1. DBSCAN聚类（基于密度）
        dbscan_dict = apply_clustering_and_visualize(
            "DBSCAN",
            DBSCAN(eps=150, min_samples=2, metric='precomputed'),
            "DBSCAN聚类结果"
        )

        # 2. 层次聚类（凝聚式）- 自适应聚类数量
        n_nodes = len(G.nodes())
        n_clusters_hierarchical = min(25, max(2, n_nodes // 2))  # 确保聚类数不超过节点数的一半
        print(f"层次聚类使用 {n_clusters_hierarchical} 个聚类")
        
        hierarchical_dict = apply_clustering_and_visualize(
            "层次聚类",
            AgglomerativeClustering(
                n_clusters=n_clusters_hierarchical,
                metric="precomputed",
                linkage='average'
            ),
            "层次聚类结果"
        )

        # 3. K-means聚类（基于质心）- 自适应聚类数量
        n_clusters_kmeans = min(25, max(2, n_nodes // 2))  # 确保聚类数不超过节点数的一半
        print(f"K-means聚类使用 {n_clusters_kmeans} 个聚类")
        
        from sklearn.cluster import KMeans, kmeans_plusplus
        kmeans_dict = apply_clustering_and_visualize(
            "K-means",
            KMeans(n_clusters=n_clusters_kmeans, random_state=42, max_iter=300),
            "K-means聚类结果"
        )

        plt.show()
        print("所有聚类图片已保存到 /tmp/ 目录")

        # 分析出口节点
        all_nodes_idx = set(range(len(nodeID)))
        source_nodes_idx = set(edge_attr['from'].unique())
        sink_nodes_idx = all_nodes_idx - source_nodes_idx
        
        print(f"发现 {len(sink_nodes_idx)} 个出口节点")
        
        # 输出聚类结果
        if sink_nodes_idx:
            sink_clusters = {index_to_nodeID[idx]: hierarchical_dict.get(index_to_nodeID[idx], -1) 
                           for idx in sink_nodes_idx if index_to_nodeID[idx] in hierarchical_dict}
            print("出口节点聚类结果:", set(sink_clusters.values()))

    except Exception as e:
        print(f"聚类过程中出错: {e}")
        import traceback
        traceback.print_exc()

debug = 0

if __name__ == '__main__':
    debug = 1
    main(debug)
