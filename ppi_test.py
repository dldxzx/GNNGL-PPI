import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from lime import lime_tabular
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch.nn.functional as F

from gnn_data import GNN_DATA
from gnn_model import GNNGL_PPI
from subgraph import SubgraphsData, extract_subgraphs, to_sparse, combine_subgraphs
from utils import Metrictor_PPI


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Test Model')
parser.add_argument('--description', default=None, type=str,
                    help='train description')
parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default=None, type=str,
                    help='protein sequence vector path')
parser.add_argument('--pre_emb_path', default=None, type=str,
                    help='protein sequence pretrained emb path')
parser.add_argument('--index_path', default=None, type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--gnn_model', default=None, type=str,
                    help="gnn trained model")
parser.add_argument('--test_all', default='False', type=boolean_string,
                    help="test all or test separately")


# 提取指定层级的特征
def extract_features(model, valid_edge_id, graph, target_layer):
    activations = {}

    def hook_fn(module, input, output):
        activations[target_layer] = output.detach()

    hook = model._modules.get(target_layer).register_forward_hook(hook_fn)
    model(graph.x, graph.edge_index, valid_edge_id, graph.edge_attr, graph)
    feature_map = activations[target_layer]
    hook.remove()  # 移除钩子
    return feature_map


def normalize_correlation(tensor_subset1, tensor_subset2):
    # 计算相关性矩阵
    correlation_matrix = torch.matmul(tensor_subset1, tensor_subset2.t())

    # 将相关性矩阵标准化到 -1 到 1 范围
    normalized_correlation = 2 * (correlation_matrix - correlation_matrix.min()) / (
            correlation_matrix.max() - correlation_matrix.min()) - 1

    return normalized_correlation


# 显示特征热图
def visualize_features(feature_map, node_id, layer):
    print(feature_map.shape)

    selected_samples = feature_map[:10, :]
    # indices1 = node_id[0]
    # indices2 = node_id[1]
    indices1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    indices2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # # # 根据索引提取子张量
    tensor_subset1 = feature_map[indices1]
    # print(selected_samples)
    # print(tensor_subset1)
    # exit(0)
    tensor_subset2 = feature_map[indices2]
    correlation_matrix = normalize_correlation(tensor_subset1, tensor_subset2)

    correlation_matrix = np.corrcoef(selected_samples.detach().numpy())

    # # 取出特征图的第一个通道
    # channel = feature_map[0]
    # # 使用 seaborn 中的热图可视化
    # sns.heatmap(channel.mean(dim=0), cmap='viridis')
    sns.set(rc={'figure.figsize': (10, 8), 'figure.dpi': 200})
    heatmap = sns.heatmap(correlation_matrix, cmap='viridis', xticklabels=indices1.tolist(),
                          yticklabels=indices2.tolist())
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0, horizontalalignment='right')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    plt.show()
    plt.savefig('/home/newdisk1/mff/mul_kg/MASSA-master/Multimodal_downstream/GNN-PPI/ppi_protein_go/'
                'Result/NE/N/' + layer + '.png')
    plt.close()


def permutation_importance(model, graph, test_mask, device, metric_fn=f1_score, num_permutations=100):
    # 计算基准 F1 分数
    baseline_score = metric_fn(graph.edge_attr_1[test_mask].cpu().data, test(model, graph, test_mask, device),
                               average='micro')
    # 循环对每个特征进行随机置换并计算分数
    permuted_scores = []
    for feature_idx in range(20):
        permuted_data = graph.x.clone()
        permuted_data[:, feature_idx] = torch.rand_like(permuted_data[:, feature_idx])
        graph.x = permuted_data
        permuted_score = metric_fn(graph.edge_attr_1[test_mask].cpu().data, test(model, graph, test_mask, device),
                                   average='micro')
        permuted_scores.append(permuted_score)

    # 计算 Permutation Importance
    importance = baseline_score - np.mean(permuted_scores)
    # print(permuted_scores)
    # exit(0)

    # Calculate importance for each feature
    importances = baseline_score - np.array(permuted_scores)

    # Get the indices of the top k features with the maximum impact
    top_k_indices = np.argsort(importances)[-10:][::-1]

    top_k_importances = importances[top_k_indices]

    # Print the top k feature indices and their importances
    print(f"Top {10} features:")
    for idx in top_k_indices:
        print(f"Feature {idx}: Importance = {importances[idx]}")
    # 画出前 k 个特征的置换重要性
    plt.bar(range(len(top_k_importances)), top_k_importances)
    plt.xticks(range(len(top_k_importances)), top_k_indices)
    plt.xlabel('Feature Index')
    plt.ylabel('Permutation Importance')
    plt.show()

    # # 画出 Permutation Importance 的条形图
    # feature_names = [f'Feature {i}' for i in range(10)]
    # plt.bar(feature_names, permuted_scores)
    # plt.title('Permutation Importance')
    # plt.xlabel('Feature Index')
    # plt.ylabel('Score')
    plt.savefig(
        '/home/newdisk1/mff/mul_kg/MASSA-master/Multimodal_downstream/GNN-PPI/ppi_protein_go/Result/PRE/permutation_importance.png')
    plt.show()

    return importance





def explain_instance(model, graph, valid_edge_id, target_class, device):
    # Create a LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(graph.x.cpu().numpy(), mode="classification")

    # Get the original features for the instance
    original_features = graph.x.cpu().numpy()
    data_row = original_features[0, :]
    output = model(graph.x, graph.edge_index, valid_edge_id, graph.edge_attr, graph)
    print(data_row.shape)
    # Explain the instance
    explanation = explainer.explain_instance(data_row, output, num_features=2, top_labels=0)
    # Get the local features and their weights
    local_features, weights = explanation.as_list(target_class)

    # Plot the local feature weights
    plt.figure(figsize=(8, 6))
    sns.barplot(x=weights, y=local_features, palette="viridis")
    plt.title(f"LIME Explanation for Class {target_class}")
    plt.xlabel("Weight")
    plt.ylabel("Feature")
    plt.show()
    plt.savefig(
        '/home/newdisk1/mff/mul_kg/MASSA-master/Multimodal_downstream/GNN-PPI/ppi_protein_go/Result/LIME/1.png')
    plt.close()


def test(model, graph, test_mask, device):
    valid_pre_result_list = []
    valid_label_list = []
    model.eval()

    batch_size = 10

    valid_steps = math.ceil(len(test_mask) / batch_size)

    for step in tqdm(range(valid_steps)):
        if step == valid_steps - 1:
            valid_edge_id = test_mask[step * batch_size:]
        else:
            valid_edge_id = test_mask[step * batch_size: step * batch_size + batch_size]

        output= model(graph.x, graph.edge_index, valid_edge_id, graph.edge_attr, graph)

    valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
    valid_label_list = torch.cat(valid_label_list, dim=0)
    # exit(0)
    metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)

    metrics.show_result()
    print("Recall: {}, Precision: {}, F1: {}".format(metrics.Recall, metrics.Precision, metrics.F1))

    return valid_pre_result_list


def main():
    args = parser.parse_args()

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)

    ppi_data.get_feature_pretrained(pseq_path=args.pseq_path, pre_emb_path=args.pre_emb_path)
    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path)

    ppi_data.generate_data()

    graph = ppi_data.data

    graph = SubgraphsData(**{k: v for k, v in graph})

    # Step 2: extract subgraphs
    subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense = extract_subgraphs(graph.edge_index,
                                                                                        graph.x.shape[0],
                                                                                        num_hops=1,
                                                                                        walk_length=0, p=1,
                                                                                        q=1, repeat=5)
    subgraphs_nodes, subgraphs_edges, hop_indicator = to_sparse(subgraphs_nodes_mask, subgraphs_edges_mask,
                                                                hop_indicator_dense)

    combined_subgraphs = combine_subgraphs(graph.edge_index, subgraphs_nodes, subgraphs_edges,
                                           num_selected=graph.num_nodes, num_nodes=graph.num_nodes)
    graph.subgraphs_batch = subgraphs_nodes[0]
    graph.subgraphs_nodes_mapper = subgraphs_nodes[1]
    graph.subgraphs_edges_mapper = subgraphs_edges[1]
    graph.combined_subgraphs = combined_subgraphs
    graph.hop_indicator = hop_indicator
    graph.__num_nodes__ = graph.num_nodes  # set number of nodes of the current graph

    temp = graph.edge_index.transpose(0, 1).numpy()
    ppi_list = []

    for edge in temp:
        ppi_list.append(list(edge))

    truth_edge_num = len(ppi_list) // 2
    # fake_edge_num = len(ppi_data.fake_edge) // 2
    fake_edge_num = 0

    with open(args.index_path, 'r') as f:
        index_dict = json.load(f)
        f.close()
    graph.train_mask = index_dict['train_index']

    graph.val_mask = index_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    node_vision_dict = {}
    for index in graph.train_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 1
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 1

    for index in graph.val_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 0
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 0

    vision_num = 0
    unvision_num = 0
    for node in node_vision_dict:
        if node_vision_dict[node] == 1:
            vision_num += 1
        elif node_vision_dict[node] == 0:
            unvision_num += 1
    print("vision node num: {}, unvision node num: {}".format(vision_num, unvision_num))

    test1_mask = []
    test2_mask = []
    test3_mask = []

    for index in graph.val_mask:
        ppi = ppi_list[index]
        temp = node_vision_dict[ppi[0]] + node_vision_dict[ppi[1]]
        if temp == 2:
            test1_mask.append(index)
        elif temp == 1:
            test2_mask.append(index)
        elif temp == 0:
            test3_mask.append(index)
    print("test1 edge num: {}, test2 edge num: {}, test3 edge num: {}".format(len(test1_mask), len(test2_mask),
                                                                              len(test3_mask)))

    graph.test1_mask = test1_mask
    graph.test2_mask = test2_mask
    graph.test3_mask = test3_mask

    device = torch.device('cpu')
    model = GNNGL_PPI(graph, gin_in_feature=256, num_layers=1, hidden=512, use_jk=False, train_eps=True,
                      feature_fusion=None, class_num=7).to(device)

    model.load_state_dict(torch.load(args.gnn_model, map_location=torch.device('cpu'))['state_dict'])
    model.load_state_dict(torch.load(args.gnn_model)['state_dict'])

    graph.to(device)

    # 调用 permutation_importance
    # importance = permutation_importance(model, graph, graph.val_mask, device)
    # print("Permutation Importance:", importance)
    # exit(0)

    if args.test_all:
        print("---------------- valid-test-all result --------------------")
        test(model, graph, graph.val_mask, device)
    else:
        print("---------------- valid-test1 result --------------------")
        if len(graph.test1_mask) > 0:
            test(model, graph, graph.test1_mask, device)
        print("---------------- valid-test2 result --------------------")
        if len(graph.test2_mask) > 0:
            test(model, graph, graph.test2_mask, device)
        print("---------------- valid-test3 result --------------------")
        if len(graph.test3_mask) > 0:
            test(model, graph, graph.test3_mask, device)


if __name__ == "__main__":
    main()
