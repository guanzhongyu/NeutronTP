# prepare_coo_graph.py

import torch
import numpy as np
import os

def process_edge_list_maxnode(file_path, save_path, num_features=128, num_classes=7, device='cpu'):
    """
    将边表文件转换为 COO_Graph 格式（节点编号从0开始，最大编号+1作为节点数）

    Args:
        file_path: 边表路径，每行: src dst weight
        save_path: 输出路径（.torch 文件）
        num_features: 节点特征维度
        num_classes: 节点分类类别数
        device: 'cpu' 或 'cuda'
    """

    # 读取边表并处理异常行
    edges = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue  # 跳过空行
            if len(parts) < 2:
                continue  # 无效行
            elif len(parts) == 2:
                src, dst = parts
                w = 1.0  # 默认权重
            else:
                src, dst, w = parts[:3]
            try:
                edges.append([float(src), float(dst), float(w)])
            except ValueError:
                print(f"跳过无法解析的行: {line.strip()}")
                continue

    if len(edges) == 0:
        raise ValueError(f"文件 {file_path} 中没有有效的边数据")

    edge_list = np.array(edges, dtype=np.float32)
    src = edge_list[:, 0].astype(np.int64)
    dst = edge_list[:, 1].astype(np.int64)
    weights = edge_list[:, 2].astype(np.float32)

    # 计算节点总数
    num_nodes = int(max(src.max(), dst.max()) + 1)
    num_edges = len(src)

    # COO 邻接矩阵
    indices = np.vstack((src, dst))            # shape (2, num_edges)
    indices = torch.from_numpy(indices).long().to(device)
    values = torch.tensor(weights, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device)

    # 节点特征
    features = torch.rand((num_nodes, num_features), dtype=torch.float32, device=device)

    # 节点标签随机生成
    labels = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long, device=device)

    # train/val/test mask
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    perm = torch.randperm(num_nodes)
    train_mask[perm[:num_nodes//2]] = True
    val_mask[perm[num_nodes//2:num_nodes*3//4]] = True
    test_mask[perm[num_nodes*3//4:]] = True

    # 构建字典
    attr_dict = {
        'adj': adj,
        'edge_index': indices,
        'features': features,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_classes': num_classes
    }

    # 保存
    torch.save(attr_dict, save_path)
    print(f"Graph saved to {save_path}, num_nodes={num_nodes}, num_edges={num_edges}")

if __name__ == "__main__":
    # 边表文件
    # file_path = 'graph_data/LJ_srt_wei_cn_train.txt'
    file_path = 'graph_data/com_srt_weg_cn_train.txt'

    # save_path = 'data/LiveJournal.torch'  # 保存路径
    save_path = 'data/ComOrkut.torch'  # 保存路径

    # 运行生成函数
    process_edge_list_maxnode(file_path, save_path,
                              num_features=128,
                              num_classes=7,
                              device='cpu')