import copy
import json
import pickle
import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
from tqdm import tqdm

from utils import UnionFindSet, get_bfs_sub_graph, get_dfs_sub_graph


class GNN_DATA:
    def __init__(self, ppi_path, exclude_protein_path=None, max_len=2000, skip_head=True, p1_index=0, p2_index=1,
                 label_index=2, graph_undirection=True, bigger_ppi_path=None):
        self.ppi_list = []
        self.ppi_dict = {}
        self.ppi_label_list = []
        self.protein_dict = {}
        self.go_onehot_feature = {}
        self.go_biobert_feature = {}
        self.protein_name = {}
        self.ppi_path = ppi_path
        self.bigger_ppi_path = bigger_ppi_path
        self.max_len = max_len
        self.pro_go_edge_type = []
        self.pro_go_edge = []

        name = 0
        ppi_name = 0
        # maxlen = 0
        self.node_num = 0
        self.edge_num = 0
        if exclude_protein_path != None:
            with open(exclude_protein_path, 'r') as f:
                ex_protein = json.load(f)
                f.close()
            ex_protein = {p: i for i, p in enumerate(ex_protein)}
        else:
            ex_protein = {}

        class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5,
                     'expression': 6}
        # iiii=0
        for line in tqdm(open(ppi_path)):
            if skip_head:
                skip_head = False
                continue
            line = line.strip().split('\t')

            if line[p1_index] in ex_protein.keys() or line[p2_index] in ex_protein.keys():
                continue

            # get node and node name
            if line[p1_index] not in self.protein_name.keys():
                self.protein_name[line[p1_index]] = name
                name += 1

            if line[p2_index] not in self.protein_name.keys():
                self.protein_name[line[p2_index]] = name
                name += 1

            # get edge and its label
            temp_data = ""
            if line[p1_index] < line[p2_index]:
                temp_data = line[p1_index] + "__" + line[p2_index]
            else:
                temp_data = line[p2_index] + "__" + line[p1_index]

            if temp_data not in self.ppi_dict.keys():
                self.ppi_dict[temp_data] = ppi_name
                temp_label = [0, 0, 0, 0, 0, 0, 0]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list.append(temp_label)
                ppi_name += 1
            else:
                index = self.ppi_dict[temp_data]
                temp_label = self.ppi_label_list[index]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list[index] = temp_label


        if bigger_ppi_path != None:

            skip_head = True
            for line in tqdm(open(bigger_ppi_path)):
                if skip_head:
                    skip_head = False
                    continue
                line = line.strip().split('\t')

                if line[p1_index] not in self.protein_name.keys():
                    self.protein_name[line[p1_index]] = name
                    name += 1

                if line[p2_index] not in self.protein_name.keys():
                    self.protein_name[line[p2_index]] = name
                    name += 1

                temp_data = ""
                if line[p1_index] < line[p2_index]:
                    temp_data = line[p1_index] + "__" + line[p2_index]
                else:
                    temp_data = line[p2_index] + "__" + line[p1_index]

                if temp_data not in self.ppi_dict.keys():
                    self.ppi_dict[temp_data] = ppi_name
                    temp_label = [0, 0, 0, 0, 0, 0, 0]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list.append(temp_label)
                    ppi_name += 1
                else:
                    index = self.ppi_dict[temp_data]
                    temp_label = self.ppi_label_list[index]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list[index] = temp_label

        i = 0
        for ppi in tqdm(self.ppi_dict.keys()):
            name = self.ppi_dict[ppi]
            assert name == i
            i+=1
            temp = ppi.strip().split('__')
            self.ppi_list.append(temp)

        ppi_num = len(self.ppi_list)
        self.origin_ppi_list = copy.deepcopy(self.ppi_list)
        # exit(0)
        assert len(self.ppi_list) == len(self.ppi_label_list)

        for i in tqdm(range(ppi_num)):
            seq1_name = self.ppi_list[i][0]
            seq2_name = self.ppi_list[i][1]
            self.ppi_list[i][0] = self.protein_name[seq1_name]

            self.ppi_list[i][1] = self.protein_name[seq2_name]


        #[7344, 7345, 7346, 7348, 7349, 7350, 1611, 29803, 37022, 43449]

        # print(self.ppi_list[7344])
        # exit(0)
        # print(len(self.ppi_label_list))
        # # 统计每个位置上出现1的次数
        # counts = [sum(col) for col in zip(*self.ppi_label_list)]

        # # 输出结果
        # for i, count in enumerate(counts):
        #     print(f'在位置 {i + 1} 上出现1的次数为: {count}')
        # exit(0)
        if graph_undirection:
            for i in tqdm(range(ppi_num)):
                temp_ppi = self.ppi_list[i][::-1]
                temp_ppi_label = self.ppi_label_list[i]
                self.ppi_list.append(temp_ppi)
                self.ppi_label_list.append(temp_ppi_label)

        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ppi_list)

        # lll=[705, 1332, 699, 704, 705, 705, 705, 705, 705, 705,
        #         1230, 1230, 705, 705, 706, 711, 707, 708, 1224, 710]
        # llll=[]
        #
        # l=[['9606.ENSP00000053468', 699], ['9606.ENSP00000081029', 704], ['9606.ENSP00000196551', 705],
        #  ['9606.ENSP00000217182', 706], ['9606.ENSP00000215956', 707], ['9606.ENSP00000202773', 708],
        #  ['9606.ENSP00000216190', 710], ['9606.ENSP00000216774', 711], ['9606.ENSP00000199320', 1224],
        #  ['9606.ENSP00000258416', 1230], ['9606.ENSP00000250896', 1332]]
        #
        # for k,v in self.protein_name.items():
        #     if v in lll:
        #         llll.append([k,v])
        # print(llll)
        # exit(0)



        # lll=[1987, 1987, 1987, 1987, 1987, 1989, 1989, 1989, 1989, 1989,
        #      1993, 1995, 1996, 1997, 1998, 1993, 1992, 1991, 1990,  400]
        #
        # llll=[]
        # for k,v in self.protein_name.items():
        #     if v in lll:
        #         llll.append([k,v])
        # print(llll)
        # exit(0)




    def get_protein_aac(self, pseq_path):
        # aac: amino acid sequences

        self.pseq_path = pseq_path
        self.pseq_dict = {}
        self.protein_len = []

        for line in tqdm(open(self.pseq_path)):
            line = line.strip().split('\t')
            if line[0] not in self.pseq_dict.keys():
                self.pseq_dict[line[0]] = line[1]
                self.protein_len.append(len(line[1]))

        print("protein num: {}".format(len(self.pseq_dict)))
        print("protein average length: {}".format(np.average(self.protein_len)))
        print("protein max & min length: {}, {}".format(np.max(self.protein_len), np.min(self.protein_len)))

    def embed_normal(self, seq, dim):
        if len(seq) > self.max_len:
            return seq[:self.max_len]
        elif len(seq) < self.max_len:
            less_len = self.max_len - len(seq)
            return np.concatenate((seq, np.zeros((less_len, dim))))
        return seq

    def vectorize(self, vec_path):
        self.acid2vec = {}
        self.dim = None
        for line in open(vec_path):
            line = line.strip().split('\t')
            temp = np.array([float(x) for x in line[1].split()])
            self.acid2vec[line[0]] = temp
            if self.dim is None:
                self.dim = len(temp)
        print("acid vector dimension: {}".format(self.dim))

        self.pvec_dict = {}

        for p_name in tqdm(self.pseq_dict.keys()):
            temp_seq = self.pseq_dict[p_name]
            temp_vec = []
            for acid in temp_seq:
                temp_vec.append(self.acid2vec[acid])
            temp_vec = np.array(temp_vec)

            temp_vec = self.embed_normal(temp_vec, self.dim)

            self.pvec_dict[p_name] = temp_vec

    def pretrained_emb_init(self, pre_emb_path):
        with open(pre_emb_path, 'rb') as f:
            self.pretrained_emb_dict = pickle.load(f)
        f.close()

    # 添加经过预训练的embedding
    def get_feature_pretrained(self, pseq_path, pre_emb_path):
        self.get_protein_aac(pseq_path)
        self.pretrained_emb_init(pre_emb_path)
        for name in tqdm(self.protein_name.keys()):
            self.protein_dict[name] = np.array(
                self.pretrained_emb_dict[name.split('.')[1]])  # pretrained_emb_dict[name]: (512, )
        print('self.protein_dict', len(self.protein_dict))

    # def go_onehot(self, go_onehot_path):
    #     with open(go_onehot_path, 'rb') as f:
    #         self.go_onehot_dict = pickle.load(f)
    #
    # def get_go_onehot(self, go_onehot_path):
    #     self.go_onehot(go_onehot_path)
    #     # print(self.go_onehot_dict)
    #     for name in tqdm(self.protein_name.keys()):
    #         if name in self.go_onehot_dict.keys():
    #             self.go_onehot_feature[name] = self.go_onehot_dict[name]
    #         else:
    #             zero_array = np.zeros(170)
    #             self.go_onehot_feature[name] = zero_array

    # def go_biobert(self, go_biobert_path):
    #     with open(go_biobert_path, 'rb') as f:
    #         self.go_biobert_dict = pickle.load(f)
    #
    # def pro_go(self, pro_go_path):
    #     with open(pro_go_path, 'rb') as f:
    #         self.pro_go_dict = pickle.load(f)

    # def pro_go_def_pad(self, pro_go_def_path):
    #     with open(pro_go_def_path, 'rb') as f:
    #         self.pro_go_def_dict = pickle.load(f)
    #
    # def get_go_biobert(self, pro_go_def_path):
    #     self.pro_go_def_pad(pro_go_def_path)
    #     for name in tqdm(self.protein_name.keys()):
    #         if name in self.pro_go_def_dict.keys():
    #             self.go_biobert_feature[name] = self.pro_go_def_dict[name]
    #         else:
    #             zero_array = np.zeros([2000, 128])
    #             self.go_biobert_feature[name] = zero_array

        # print(self.pro_go_def_dict)
        # exit(0)

    # def get_graph_GO(self):
    #     str = []
    #     drc = []
    #     path = "/home/newdisk1/mff/mul_kg/MASSA-master/data/seq/SHS27K/go_nan_string.pkl"
    #     f = open(path, 'rb')
    #     protein_go_dict = pickle.load(f)
    #     for i in range(len(self.origin_ppi_list)):
    #         same_edge_count = []
    #         if self.origin_ppi_list[i][0] in protein_go_dict.keys() and self.origin_ppi_list[i][
    #             1] in protein_go_dict.keys():
    #             edges1 = protein_go_dict[self.origin_ppi_list[i][0]]
    #             edges2 = protein_go_dict[self.origin_ppi_list[i][1]]
    #             for e1, e2 in zip(edges1, edges2):
    #                 if e1 == 1 and e2 == 1:
    #                     same_edge_count.append(1)
    #                 if e1 == 1 and e2 == 0:
    #                     same_edge_count.append(0)
    #                 if e2 == 1 and e1 == 0:
    #                     same_edge_count.append(0)
    #                 elif e1 == 0 and e2 == 0:
    #                     same_edge_count.append(0)
    #         else:
    #             same_edge_count = [0, 0, 0]
    #         self.pro_go_edge_type.append(same_edge_count)

    def get_feature_origin(self, pseq_path, vec_path):
        self.get_protein_aac(pseq_path)
        self.vectorize(vec_path)
        self.protein_dict_origin = {}
        for name in tqdm(self.protein_name.keys()):
            self.protein_dict_origin[name] = self.pvec_dict[name]  # pvec_dict[name]: (2000, 13)

    def get_connected_num(self):
        self.ufs = UnionFindSet(self.node_num)
        ppi_ndary = np.array(self.ppi_list)
        for edge in ppi_ndary:
            start, end = edge[0], edge[1]
            self.ufs.union(start, end)

    def generate_data(self):
        self.get_connected_num()

        print("Connected domain num: {}".format(self.ufs.count))

        # print(self.ppi_label_list[:5])
        self.ppi_label_list_tup = [tuple(label) for label in self.ppi_label_list]
        set_ppi_label = list(set(self.ppi_label_list_tup))
        A = {}
        for i in range(len(set_ppi_label)):
            A[set_ppi_label[i]] = i
        self.mul_type = []
        for i in range(len(self.ppi_label_list)):
            self.mul_type.append(A[tuple(self.ppi_label_list[i])])

        mul_type = np.array(self.mul_type)
        self.mul_type = torch.tensor(mul_type)

        ppi_list = np.array(self.ppi_list)
        ppi_label_list = np.array(self.ppi_label_list)

        self.edge_index = torch.tensor(ppi_list, dtype=torch.long)
        self.edge_attr = torch.tensor(ppi_label_list, dtype=torch.long)

        # for i in range(len(self.pro_go_edge_type)):
        #     temp_edge_go_onehot = self.pro_go_edge_type[i][::-1]
        #     self.pro_go_edge_type.append(temp_edge_go_onehot)
        # pro_go_edge_type = np.array(self.pro_go_edge_type)
        # self.pro_go_edge_attr = torch.tensor(pro_go_edge_type)

        self.x = []
        i = 0
        for name in self.protein_name:
            assert self.protein_name[name] == i
            i += 1
            self.x.append(self.protein_dict[name])

        self.x = np.array(self.x)
        self.x = torch.tensor(self.x, dtype=torch.float)

        # self.x_go_onehot = []
        # i = 0
        # for name in self.protein_name:
        #     assert self.protein_name[name] == i
        #     i += 1
        #     self.x_go_onehot.append(self.go_onehot_feature[name])
        #
        # self.x_go_onehot = np.array(self.x_go_onehot)
        # self.x_go_onehot = torch.tensor(self.x_go_onehot, dtype=torch.float)

        # self.x_go_biobert = []
        # i = 0
        # for name in self.protein_name:
        #     assert self.protein_name[name] == i
        #     i += 1
        #     self.x_go_biobert.append(self.go_biobert_feature[name])
        #
        # self.x_go_biobert = np.array(self.x_go_biobert)
        # self.x_go_biobert = torch.tensor(self.x_go_biobert, dtype=torch.float)

        self.x_origin = []
        i = 0
        for name in self.protein_name:
            assert self.protein_name[name] == i
            i += 1
            self.x_origin.append(self.protein_dict_origin[name])

        self.x_origin = np.array(self.x_origin)
        self.x_origin = torch.tensor(self.x_origin)

        # graph_data = Data(x=self.x, edge_index=self.edge_index.T)
        # nx_graph1 = to_networkx(graph_data)
        # for node in nx_graph1.nodes():
        #     nx_graph1.nodes[node]['x'] = graph_data.x[node].numpy()
        #
        # nx_graph2 = to_networkx(graph_data)
        # for node in nx_graph2.nodes():
        #     nx_graph2.nodes[node]['x'] = graph_data.x[node].numpy()
        #
        # nx_graph3 = to_networkx(graph_data)
        # for node in nx_graph3.nodes():
        #     nx_graph3.nodes[node]['x'] = graph_data.x[node].numpy()

        # 在NetworkX图上随机删除节点
        # nodes_to_remove = random.sample(list(nx_graph1.nodes()), k=int(0.1 * nx_graph1.number_of_nodes()))
        # nx_graph1.remove_nodes_from(nodes_to_remove)
        #
        # # 在NetworkX图上随机删除边
        # edges_to_remove = random.sample(list(nx_graph2.edges()), k=int(0.1 * nx_graph2.number_of_edges()))
        # nx_graph2.remove_edges_from(edges_to_remove)
        #
        # # 随机添加一些边
        # num_edges_to_add = int(0.1 * nx_graph2.number_of_edges())
        # nodes = list(nx_graph3.nodes())
        # for _ in range(num_edges_to_add):
        #     node1 = random.choice(nodes)
        #     node2 = random.choice(nodes)
        #     # 确保不添加自环边和重复边
        #     while node1 == node2 or nx_graph3.has_edge(node1, node2):
        #         node1 = random.choice(nodes)
        #         node2 = random.choice(nodes)
        #     nx_graph3.add_edge(node1, node2)

        # # 属性掩盖
        # for node in nx_graph4.nodes():
        #     if random.random() > 0.5:
        #         nx_graph4.nodes[node]['x'] = np.zeros(512)

        # 将NetworkX图转换回PyG图数据对象
        # self.pyg_graph_data1 = from_networkx(nx_graph1)
        # self.pyg_graph_data2 = from_networkx(nx_graph2)
        # self.pyg_graph_data3 = from_networkx(nx_graph3)
        # self.pyg_graph_data4 = from_networkx(nx_graph4)
        #
        # print(self.pyg_graph_data1)
        # print(self.pyg_graph_data2)
        # print(self.pyg_graph_data3)
        # print(self.pyg_graph_data4)
        # exit(0)

        self.data = Data(x=self.x, x_origin=self.x_origin, edge_index=self.edge_index.T, edge_attr_1=self.edge_attr,
                         edge_mul=self.mul_type)

    def split_dataset(self, train_valid_index_path, test_size=0.2, random_new=False, mode='random'):
        if random_new:
            if mode == 'random':
                print('+++++++++++++++++++++++++++++random++++++++++++++++++++++++++++++++++++++')
                ppi_num = int(self.edge_num // 2)
                random_list = [i for i in range(ppi_num)]
                random.shuffle(random_list)

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = random_list[: int(ppi_num * (1 - test_size))]
                self.ppi_split_dict['valid_index'] = random_list[int(ppi_num * (1 - test_size)):]

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            elif mode == 'bfs' or mode == 'dfs':
                print("use {} methed split train and valid dataset".format(mode))
                node_to_edge_index = {}
                edge_num = int(self.edge_num // 2)
                for i in range(edge_num):
                    edge = self.ppi_list[i]
                    if edge[0] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[0]] = []
                    node_to_edge_index[edge[0]].append(i)

                    if edge[1] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[1]] = []
                    node_to_edge_index[edge[1]].append(i)

                node_num = len(node_to_edge_index)

                sub_graph_size = int(edge_num * test_size)
                if mode == 'bfs':
                    print('+++++++++++++++++++++++++++++bfs++++++++++++++++++++++++++++++++++++++')
                    selected_edge_index = get_bfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)
                elif mode == 'dfs':
                    print('+++++++++++++++++++++++++++++dfs++++++++++++++++++++++++++++++++++++++')
                    selected_edge_index = get_dfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)

                all_edge_index = [i for i in range(edge_num)]

                unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = unselected_edge_index
                self.ppi_split_dict['valid_index'] = selected_edge_index

                assert len(unselected_edge_index) + len(selected_edge_index) == edge_num

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            else:
                print("your mode is {}, you should use bfs, dfs or random".format(mode))
                return
        else:
            with open(train_valid_index_path, 'r') as f:
                self.ppi_split_dict = json.load(f)
                f.close()
