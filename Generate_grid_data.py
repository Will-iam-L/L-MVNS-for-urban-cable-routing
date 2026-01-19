import matplotlib
matplotlib.use('TkAgg')  # 必须在其他 matplotlib 导入之前调用
import random
import networkx as nx
from sklearn.cluster import KMeans
import pickle
import math
import numpy as np




def generate_regular_grid(rows, cols, width=1, height=1):
    G = nx.Graph()

    # 计算每个节点的位置
    for r in range(rows):
        for c in range(cols):
            # 计算当前节点的 x, y 坐标
            x_pos = c * width  # 固定的宽度
            y_pos = -r * height  # 固定的高度 (取负以显示在正确方向)
            G.add_node((r, c), pos=(x_pos, y_pos))  # 添加节点并指定位置

            # 添加与右侧节点的边（如果存在）
            if c < cols - 1:
                G.add_edge((r, c), (r, c + 1), weight=100.0)  # 添加权重参数

            # 添加与下方节点的边（如果存在）
            if r < rows - 1:
                G.add_edge((r, c), (r + 1, c), weight=100.0)

    return G


def cluster_points(G, num_clusters_red, num_clusters_black):
    """根据网格节点的随机值生成聚类后的红色点"""
    positions = np.array([G.nodes[node]['pos'] for node in G.nodes()])
    values = np.ones(len(G.nodes()))  # 生成随机值

    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters=num_clusters_red)
    kmeans.fit(positions, sample_weight=values)  # 使用随机值作为样本权重

    # 生成聚类中心作为红色节点
    red_nodes = kmeans.cluster_centers_
    red_centers = [tuple(center) for center in red_nodes]

    # 对红色节点生成聚类中心作为黑色节点
    kmeans_black = KMeans(n_clusters=num_clusters_black)
    kmeans_black.fit(red_nodes)
    black_nodes = kmeans_black.cluster_centers_
    black_centers = [tuple(center) for center in black_nodes]

    # 返回红色节点的坐标，转换为可以哈希的元组
    return red_centers, black_centers


def closest_point_on_edge(point, edge):
    """返回给定点在边上的最近点"""
    start_pos = np.array(edge[0])
    end_pos = np.array(edge[1])

    # 计算边的方向
    edge_vector = end_pos - start_pos
    edge_length_squared = np.dot(edge_vector, edge_vector)

    if edge_length_squared == 0:  # 检查边的长度
        return start_pos  # 边的长度为0，返回端点

    # 计算投影比例
    t = np.dot(np.array(point) - start_pos, edge_vector) / edge_length_squared
    t = max(0, min(1, t))  # 限制 t 的范围在 [0, 1]

    closest_point = start_pos + t * edge_vector
    return closest_point


def map_nodes_to_edges(G, red_nodes):
    """将节点映射到最近的图的边上"""
    mapped_red_nodes = []

    for red_node in red_nodes:
        # 查找最近的边
        nearest_edge = None
        min_distance = float('inf')

        for edge in G.edges():
            edge_start = G.nodes[edge[0]]['pos']
            edge_end = G.nodes[edge[1]]['pos']
            point_on_edge = closest_point_on_edge(red_node, (edge_start, edge_end))
            distance = np.linalg.norm(np.array(red_node) - point_on_edge)

            # 更新最近边
            if distance < min_distance:
                min_distance = distance
                nearest_edge = edge

        # 计算最近边上的点
        mapped_red_node = closest_point_on_edge(red_node,
                                                (G.nodes[nearest_edge[0]]['pos'], G.nodes[nearest_edge[1]]['pos']))
        # 转换成int类型
        mapped_red_nodes.append((tuple(mapped_red_node.astype(int)), nearest_edge))  # 存储映射后的节点和对应的边

    return mapped_red_nodes


def generate_rand_data(num_grid, num_substations_10, num_substations_110):
    # 设置seed
    random.seed(42)

    for case_repeat in range(3):
        # 设置网格的行和列数
        rows = num_grid
        cols = num_grid
        # 生成网格
        G = generate_regular_grid(rows, cols, width=100, height=100)
        pos = nx.get_node_attributes(G, 'pos')
        # 生成红色节点的聚类
        red_nodes, black_nodes = cluster_points(G, num_clusters_red=num_substations_10,
                                                num_clusters_black=num_substations_110)

        # 映射红色节点到最近的边
        mapped_red_nodes_with_edges = map_nodes_to_edges(G, red_nodes)
        # 添加映射后的红色节点到 G，并找出所在的边
        for mapped_red_node, edge in mapped_red_nodes_with_edges:
            G.add_node(mapped_red_node, pos=mapped_red_node)  # 将节点作为元组添加
            # 找出映射后的红色节点所在的边
            if edge is not None:
                # 添加边
                G.add_edge(mapped_red_node, edge[0],
                           weight=math.sqrt(sum((x - y) ** 2 for x, y in zip(mapped_red_node, pos[edge[0]]))))  # 与边的起始节点连接
                G.add_edge(mapped_red_node, edge[1],
                           weight=math.sqrt(sum((x - y) ** 2 for x, y in zip(mapped_red_node, pos[edge[1]]))))  # 与边的起始节点连接
            # 更新 pos 字典
            pos[mapped_red_node] = mapped_red_node  # 更新 pos 字典
        mapped_red_x, mapped_red_y = zip(*[node for node, _ in mapped_red_nodes_with_edges])

        identified_targets_red = [node[0] for node in mapped_red_nodes_with_edges]  # 获取目标节点

        # 映射黑色节点到最近的边
        mapped_black_nodes_with_edges = map_nodes_to_edges(G, black_nodes)
        # 添加映射后的红色节点到 G，并找出所在的边
        for mapped_black_node, edge in mapped_black_nodes_with_edges:
            G.add_node(mapped_black_node, pos=mapped_black_node)  # 将节点作为元组添加
            # 找出映射后的红色节点所在的边
            if edge is not None:
                # 添加边
                G.add_edge(mapped_black_node, edge[0],
                           weight=math.sqrt(
                               sum((x - y) ** 2 for x, y in zip(mapped_black_node, pos[edge[0]]))))  # 与边的起始节点连接
                G.add_edge(mapped_black_node, edge[1],
                           weight=math.sqrt(sum((x - y) ** 2 for x, y in zip(mapped_black_node, pos[edge[1]]))))
            # 更新 pos 字典
            pos[mapped_black_node] = mapped_black_node  # 更新 pos 字典
        mapped_black_x, mapped_black_y = zip(*[node for node, _ in mapped_black_nodes_with_edges])
        identified_targets_black = [node[0] for node in mapped_black_nodes_with_edges]  # 获取目标节点

        # 绘制电缆规划路径
        max_capacity = 10  # 设定单条电缆的最大容量
        loads = {node: random.randint(2, 5) for node in identified_targets_red}

        # 将变量存储起来
        data = [G, identified_targets_red, identified_targets_black, max_capacity, pos, loads, mapped_red_x, mapped_red_y,
                mapped_black_x, mapped_black_y]

        # 保存数据到文件pkl
        with open("Data/New_Data/regular_grid_data_row{}_col{}_mv{}_hv{}_case{}.pkl".format(rows, cols, num_substations_10,
                                                                               num_substations_110, case_repeat+1), "wb") as f:
            pickle.dump(data, f)


# 主函数
if __name__ == "__main__":
    # 生成随机数据[20, 20, 4] , [20, 30, 5] ,[30, 50, 6], [30, 80, 7], [30, 100, 8] ,[40, 150, 9], [40, 200, 10]
    for num_grid, num_substations_10, num_substations_110 in [[20, 30, 5] ,[30, 50, 6], [30, 80, 7], [30, 100, 8]]:
        generate_rand_data(num_grid, num_substations_10, num_substations_110)