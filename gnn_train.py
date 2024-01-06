import argparse
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn

from asl import AsymmetricLossOptimized
from gnn_data import GNN_DATA
from gnn_model import GNNGL_PPI
from subgraph import SubgraphsData, extract_subgraphs, to_sparse, combine_subgraphs
from utils import Metrictor_PPI, print_file

# from tensorboardX import SummaryWriter

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Train Model')
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
parser.add_argument('--go_onehot_path', default=None, type=str,
                    help='protein sequence pretrained emb path')
# parser.add_argument('--go_biobert_path', default=None, type=str,
#                     help='protein sequence pretrained emb path')
parser.add_argument('--pro_go_def_path', default=None, type=str,
                    help='protein sequence pretrained emb path')

parser.add_argument('--split_new', default=None, type=boolean_string,
                    help='split new index file or not')
parser.add_argument('--split_mode', default=None, type=str,
                    help='split method, random, bfs or dfs')
parser.add_argument('--train_valid_index_path', default=None, type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--use_lr_scheduler', default=None, type=boolean_string,
                    help="train use learning rate scheduler or not")
parser.add_argument('--save_path', default=None, type=str,
                    help='model save path')
parser.add_argument('--graph_only_train', default=None, type=boolean_string,
                    help='train ppi graph conctruct by train or all(train with test)')
parser.add_argument('--batch_size', default=None, type=int,
                    help="gnn train batch size, edge batch size")
parser.add_argument('--epochs', default=None, type=int,
                    help='train epoch number')


def train(model, graph, ppi_list, loss_fn, loss_asl,
          optimizer, device,
          result_file_path, save_path,
          batch_size=512, epochs=1000, scheduler=None,
          got=False):
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0

    truth_edge_num = graph.edge_index.shape[1] // 2

    for epoch in range(epochs):

        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0
        tensor_list=[]
        label_list=[]

        steps = math.ceil(len(graph.train_mask) / batch_size)

        model.train()

        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)

        for step in range(steps):
            if step == steps - 1:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size:]
                else:
                    train_edge_id = graph.train_mask[step * batch_size:]
            else:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size: step * batch_size + batch_size]
                else:
                    train_edge_id = graph.train_mask[step * batch_size: step * batch_size + batch_size]

            if got:
                output = model(graph.x, graph.edge_index_got, train_edge_id, graph.edge_go_got, graph)
                label = graph.edge_attr_got[train_edge_id]

            else:

                output= model(graph.x, graph.edge_index, train_edge_id, graph.edge_attr, graph)
                # output = model(train_edge_id, graph)
                label = graph.edge_attr_1[train_edge_id]

            label = label.type(torch.FloatTensor).to(device)

            # loss = loss_fn(output, label)

            loss_a = loss_asl(output, label)

            # loss = loss+0.001*loss_a
            loss = loss_a

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data)

            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()

            global_step += 1
            print_file("epoch: {}, step: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}"
                       .format(epoch, step, loss.item(), metrics.Precision, metrics.Recall, metrics.F1))
            tensor_list.append(output)
            label_list.append(label)

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   os.path.join(save_path, 'gnn_model_train.ckpt'))

        valid_pre_result_list = []
        valid_label_list = []
        valid_loss_sum = 0.0

        model.eval()

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps - 1:
                    valid_edge_id = graph.val_mask[step * batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]

                output= model(graph.x, graph.edge_index, valid_edge_id, graph.edge_attr, graph)
                # output = model(valid_edge_id, graph)
                label = graph.edge_attr_1[valid_edge_id]
                label = label.type(torch.FloatTensor).to(device)

                # loss = loss_fn(output, label)
                loss_a = loss_asl(output, label)
                # loss=loss+0.001*loss_a
                loss = loss_a
                # l2_loss = mahalanobis_distances.pow(2).sum()
                valid_loss_sum += loss.item()
                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)
        metrics.show_result()
        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps

        if scheduler != None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']),
                       save_file_path=result_file_path)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(save_path, 'gnn_model_valid_best.ckpt'))

        print_file(
            "epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, Best valid_f1: {}, in {} epoch"
                .format(epoch, loss, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision, metrics.F1,
                        global_best_valid_f1, global_best_valid_f1_epoch), save_file_path=result_file_path)


def main():
    args = parser.parse_args()

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)
    # ppi_data.get_graph_GO()

    print("+++++++++++++++++++++++use_get_feature_pretrained++++++++++++++++++++++++++++++++")
    ppi_data.get_feature_pretrained(pseq_path=args.pseq_path, pre_emb_path=args.pre_emb_path)
    print("+++++++++++++++++++++++use_get_feature_vec+++++++++++++++++++++++++++++++++")
    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path)

    # ppi_data.get_go_onehot(go_onehot_path=args.go_onehot_path)
    # ppi_data.get_go_biobert(pro_go_def_path=args.pro_go_def_path)

    ppi_data.generate_data()

    print("----------------------- start split train and valid index -------------------")
    print("whether to split new train and valid index file, {}".format(args.split_new))
    if args.split_new:
        print("use {} method to split".format(args.split_mode))
    ppi_data.split_dataset(args.train_valid_index_path, random_new=args.split_new, mode=args.split_mode)
    print("----------------------- Done split train and valid index -------------------")

    graph = ppi_data.data

    print(graph)

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

    ppi_list = ppi_data.ppi_list

    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    graph.edge_index_got = torch.cat(
        (graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.train_mask][[1, 0]]), dim=1)
    graph.edge_attr_got = torch.cat((graph.edge_attr_1[graph.train_mask], graph.edge_attr_1[graph.train_mask]), dim=0)
    graph.edge_mul_type_got = torch.cat((graph.edge_mul[graph.train_mask], graph.edge_mul[graph.train_mask]), dim=0)

    # graph.edge_go_got = torch.cat((graph.edge_attr[graph.train_mask], graph.edge_attr[graph.train_mask]), dim=0)

    graph.train_mask_got = [i for i in range(len(graph.train_mask))]

    #  计算Mahalanobis距离
    # print(graph.edge_attr_got.shape)

    # 计算标签的均值
    # label_means = graph.edge_attr_got.float().mean(dim=0)
    #
    # # 计算标签之间的协方差矩阵
    # label_covariance = torch.mm((graph.edge_attr_got - label_means).t(), (graph.edge_attr_got - label_means)) / (
    #         graph.edge_attr_got.size(0) - 1)
    #
    # # 标准化协方差矩阵为相关性矩阵
    # label_correlation = label_covariance / (
    #         torch.sqrt(torch.diag(label_covariance))[:, None] * torch.sqrt(torch.diag(label_covariance))[None, :])
    #
    # # print("标签均值:")
    # # print(label_means)
    # # print("\n标签协方差矩阵:")
    # # print(label_covariance)
    # # print("\n标签相关性矩阵:")
    # # print(label_correlation)
    # # print(label_correlation.shape)
    # print(label_covariance)
    # graph.label_corr=label_covariance.cuda()
    # exit(0)

    # mahalanobis_distances = torch.sqrt(torch.diagonal(
    #     torch.mm(torch.mm(graph.edge_attr_got.float() - label_means, torch.inverse(label_correlation)),
    #              (graph.edge_attr_got.float() - label_means).t())))
    # print("\nMahalanobis距离:")
    # print(mahalanobis_distances)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    graph.to(device)

    # model = GIN_SubGraph_Net(in_len=2000, in_feature=13, gin_in_feature=512, num_layers=1, pool_size=3, cnn_hidden=1).to(
    #     device)  # origin: gin_in_feature = 256
    # model = GIN_Net_MUSE(in_len=2000, in_feature=13, gin_in_feature=512, num_layers=1, pool_size=3, cnn_hidden=1).to(
    #     device)  # origin: gin_in_feature = 256

    model = GNNGL_PPI(graph, gin_in_feature=256, num_layers=1, hidden=512, use_jk=False, train_eps=True,
                      feature_fusion=None, class_num=7).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    scheduler = None
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                               verbose=True)

    loss_fn = nn.BCEWithLogitsLoss().to(device)
    loss_asl = AsymmetricLossOptimized(gamma_neg=1, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True).to(
        device)

    # bagging_clf = BaggingClassifier(base_estimator=model,n_estimators=2)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    save_path = os.path.join(args.save_path, "gnn_{}_{}".format(args.description, time_stamp))
    result_file_path = os.path.join(save_path, "valid_results.txt")
    config_path = os.path.join(save_path, "config.txt")
    os.mkdir(save_path)

    with open(config_path, 'w') as f:
        args_dict = args.__dict__
        for key in args_dict:
            f.write("{} = {}".format(key, args_dict[key]))
            f.write('\n')
        f.write('\n')
        f.write("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    train(model, graph, ppi_list, loss_fn, loss_asl,
          optimizer, device,
          result_file_path, save_path,
          batch_size=args.batch_size, epochs=args.epochs, scheduler=scheduler,
          got=args.graph_only_train)


if __name__ == "__main__":
    main()
