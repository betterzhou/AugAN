import scipy.sparse as sp
import scipy.io
import numpy as np
import networkx as nx
import random, copy
import dgl


def load_raw_data(root_path, data_source):
    data = scipy.io.loadmat(root_path + "{}.mat".format(data_source))
    adj_csr_matrix = sp.csr_matrix(data["Network"])
    Graph = nx.from_scipy_sparse_matrix(adj_csr_matrix)

    attributes = sp.csr_matrix(data["Attributes"])
    attri_matrix = attributes.todense()
    overall_norm_attributes = overall_normalization( np.array(attri_matrix) )

    label = data["Label"]
    labels = label.reshape(-1, label.shape[0])[0]

    return overall_norm_attributes, adj_csr_matrix, labels, Graph


def overall_normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def modify_label_func(label):
    new_y = []
    for j in label:
        if j == 0:
            new_y.append(0)
        if j == 1:
            new_y.append(1)
    return np.array(new_y)


def find_all_1234_hop_Neighbor(G, this_node, self_loop=False):
    """
    input: a node
    output: a node's K-hop neighbors
    """
    nodes = list(nx.nodes(G))
    nei1_list = []
    nei2_list = []
    nei3_list = []
    nei4_list = []

    #  only 1th-hop neighbors
    for FNs in list(nx.neighbors(G, this_node)):
        nei1_list.append(FNs)

    # only 2th-hop neighbors
    for n1 in nei1_list:
        for SNs in list(nx.neighbors(G, n1)):
            nei2_list.append(SNs)
    # remove duplicates
    nei2_list = list(set(nei2_list))
    if not self_loop and this_node in nei2_list:
        nei2_list.remove(this_node)

    # only 3th-hop neighbors
    for n2 in nei2_list:
        for TNs in nx.neighbors(G, n2):
            nei3_list.append(TNs)
    # remove duplicates
    nei3_list = list(set(nei3_list))
    if not self_loop and this_node in nei3_list:
        nei3_list.remove(this_node)

    # only 4th-hop neighbors
    for n3 in nei3_list:
        for next_node in nx.neighbors(G, n3):
            nei4_list.append(next_node)
    # remove duplicates
    nei4_list = list(set(nei4_list))
    if not self_loop and this_node in nei4_list:
        nei4_list.remove(this_node)
        # -------------------------------------
    # combine all k th-hop neighbors
    all_neighbors_1_hop = nei1_list

    all_neighbors_2_hop = []
    all_neighbors_2_hop.extend(nei1_list)
    all_neighbors_2_hop.extend(nei2_list)

    nei3_list.extend(all_neighbors_2_hop)
    all_neighbors_3_hop = list(set(nei3_list))

    all_neighbors_4_hop = []
    nei4_list.extend(all_neighbors_3_hop)
    all_neighbors_4_hop = list(set(nei4_list))

    return all_neighbors_1_hop, all_neighbors_2_hop, all_neighbors_3_hop, all_neighbors_4_hop


def find_specific_class_nodes(assigned_class_list, org_labels):
    """
    :param assigned_class_list:   e.g., [1, 3]
    :param org_labels:            e.g., [1,4,7,2,6...]
    :return:  all node indices in the assigned_classes
    """
    all_nodes_list = []
    for class_j in assigned_class_list:
        class_j_nodes_indx = np.where(org_labels == class_j)[0]
        all_nodes_list.extend(list(class_j_nodes_indx))
    return np.array(all_nodes_list)


def get_global_normal_abnormal_indices(org_labels, outlier_class_list):
    """
    :param org_labels:              e.g., [1,4,7,2,6...]
    :param outlier_class_list:      e.g., [1, 3]
    :return:
    """
    # 1. get all node indices in the assigned_classes
    all_outliers_indices = find_specific_class_nodes(outlier_class_list, org_labels)

    # 2. get the rest normal index
    all_nodes_num = org_labels.shape[0]
    all_indices = np.array([i for i in range(all_nodes_num)])
    all_normal_set = set(all_indices).difference(set(all_outliers_indices))
    all_normal_indices = np.array(list(all_normal_set))

    # 3. build gnd_binary
    label_binary = np.zeros((all_nodes_num, 1))
    label_binary[all_outliers_indices] = 1
    gnds_binary = modify_label_func(label_binary)  # [[0] [1] [0]] --> [0 1 0]

    return all_normal_indices, all_outliers_indices, gnds_binary


def shuffle_array(input_array, seed):
    random.seed(seed)
    random.shuffle(input_array)
    return input_array


def Graph_connection_analysis(networkx_G):
    print('whether G is connected:', nx.is_connected(networkx_G))
    print('number_connected_components:', nx.number_connected_components(networkx_G))
    whether_fully_connected = True
    all_components_list = [list(G_i) for G_i in nx.connected_components(networkx_G)]
    if len(all_components_list) > 1:
        whether_fully_connected = False
    return whether_fully_connected


def delete_non_connected_nodes_from_G(networkx_G):
    """
    :return: a connected networkx_object
    """
    all_components_list = [list(G_i) for G_i in nx.connected_components(networkx_G)]
    sorted_component_lists = sorted(all_components_list, key=lambda x: len(x), reverse=True)
    delete_nodes_list = []
    for list_j in sorted_component_lists[1:]:
        delete_nodes_list.extend(list_j)
    preserved_nodes = sorted_component_lists[0]
    for node_j in delete_nodes_list:
        networkx_G.remove_node(node_j)
    return networkx_G, delete_nodes_list, preserved_nodes


def split_graph(org_G, nodes_list, org_attribute_mat, org_label):
    sub_Graph = dgl.node_subgraph(org_G, nodes_list)
    original_node_indx = sub_Graph.ndata[dgl.NID].numpy()
    current_index = sub_Graph.nodes().numpy()
    id_mapping_dict = dict(zip(current_index, original_node_indx))
    new_attribute_mat = org_attribute_mat[original_node_indx]
    new_labels = org_label[original_node_indx]
    assert len(list(nx.nodes(sub_Graph))) == new_attribute_mat.shape[0]
    nx_subgraph = sub_Graph.to_networkx().to_undirected()
    connected_Flag = Graph_connection_analysis(nx_subgraph)
    if connected_Flag == True:
        new_adj_sparse = nx.adjacency_matrix(nx_subgraph)
        return new_adj_sparse, new_attribute_mat, new_labels, original_node_indx
    else:
        nx_subgraph_final, deleted_node_indx, _ = delete_non_connected_nodes_from_G(nx_subgraph)
        org_node_index_final = list(copy.deepcopy(original_node_indx))
        for k in deleted_node_indx:
            org_node_index_final.remove(id_mapping_dict[k])

        new_attribute_mat_final = org_attribute_mat[org_node_index_final]
        new_labels_final = org_label[org_node_index_final]
        new_adj_sparse_final = nx.adjacency_matrix(nx_subgraph_final)

        connected_Flag_final = Graph_connection_analysis(nx_subgraph_final)
        if connected_Flag_final:
            print('ok, G is connected!')
            print(' sub_G total nodes:', new_attribute_mat_final.shape[0])
            print(' sub_G total edges:', nx_subgraph_final.number_of_edges())
            print(' sub_G attributes dim:', new_attribute_mat_final.shape[1])
        return new_adj_sparse_final, new_attribute_mat_final, new_labels_final, org_node_index_final


def split_all_rare_categories(abnormal_classes_list, org_labels, split_subG_num, random_seed):
    all_outliers_indx = find_specific_class_nodes(abnormal_classes_list, org_labels)
    all_outliers_num = all_outliers_indx.shape[0]
    outlier_num_per_subgraph = int(all_outliers_num / (split_subG_num))
    sequential_list = [i for i in range(all_outliers_num)]
    split_outlier_dict = {}
    for j in range(split_subG_num):
        local_indx_j = sequential_list[outlier_num_per_subgraph*j: outlier_num_per_subgraph*(j+1)]
        split_outlier_dict[str(j)] = all_outliers_indx[np.array(local_indx_j)]
    return split_outlier_dict


def find_max_local_index(input_distance_list):
    max_value = max(input_distance_list)
    max_local_index = input_distance_list.index(max_value)
    return max_local_index


#######################################################################################
#######################################################################################
"""
Purpose:        split Graph into sub-graphs;
"""
#######################################################################################
#######################################################################################


data_source = 'Amazon_clothing'
abnormal_classes_list = [27, 60, 31, 34, 32, 76, 5, 63, 71]
split_subG_num = 4
if data_source == 'Photo':
    abnormal_classes_list = [0, 7]
    split_subG_num = 3
if data_source == 'ACM':
    abnormal_classes_list = [2, 6]
    split_subG_num = 3
if data_source == 'DBLP':
    abnormal_classes_list = [135,24,95,106,30,131,128,43,32,94,7,53,46,62,27]
    split_subG_num = 4
if data_source == 'CS':
    abnormal_classes_list = [6, 9, 12]
    split_subG_num = 4


seeds = 133
extension_ratio = 0.5        # adjust it to control sub-graph size
random_anchor_index = 9

raw_data_path = './'
save_data_path = './sub_G_datasets/'
attri_matrix, adj_csr_matrix, org_labels, nx_Graph = load_raw_data(raw_data_path, data_source)
_, _, gnd_binary = get_global_normal_abnormal_indices(org_labels, abnormal_classes_list)

connected_Flag = Graph_connection_analysis(nx_Graph)
assert connected_Flag == False
nx_largest_subG, deleted_node_indx, preserved_nodes_index = delete_non_connected_nodes_from_G(nx_Graph)

# ---------------------------  1. find anchors -------------------------------------
largest_subG_node_list = list(nx_largest_subG.nodes())
distance_list_anchor_org = []
distance_list_anchor_b = []
distance_list_anchor_c = []
# find 2th anchor
for node_j in list(nx_largest_subG.nodes()):
    shortest_path_k = nx.shortest_path(nx_largest_subG, source=random_anchor_index, target=node_j)
    distance_list_anchor_org.append(len(shortest_path_k))
anchor_index_b_local_index = find_max_local_index(distance_list_anchor_org)
anchor_index_b = largest_subG_node_list[anchor_index_b_local_index]
print('anchor_index_b', anchor_index_b)
# find 3th anchor
for node_i in list(nx_largest_subG.nodes()):
    shortest_path_j = nx.shortest_path(nx_largest_subG, source=anchor_index_b, target=node_i)
    distance_list_anchor_b.append(len(shortest_path_j))
assert len(distance_list_anchor_org) == len(distance_list_anchor_b)
distance_list_anchor_b_avg = [distance_list_anchor_org[i] + distance_list_anchor_b[i] for i in range(len(distance_list_anchor_b))]
anchor_index_c_local_index = find_max_local_index(distance_list_anchor_b_avg)
anchor_index_c = largest_subG_node_list[anchor_index_c_local_index]
print('anchor_index_c', anchor_index_c)
# find 4th anchor
for node_i in list(nx_largest_subG.nodes()):
    shortest_path_j = nx.shortest_path(nx_largest_subG, source=anchor_index_c, target=node_i)
    distance_list_anchor_c.append(len(shortest_path_j))
assert len(distance_list_anchor_org) == len(distance_list_anchor_c)
distance_list_anchor_c_avg = [distance_list_anchor_org[i] + distance_list_anchor_b[i] + distance_list_anchor_c[i]  for i in range(len(distance_list_anchor_b))]
anchor_index_d_local_index = find_max_local_index(distance_list_anchor_c_avg)
anchor_index_d = largest_subG_node_list[anchor_index_d_local_index]
print('anchor_index_d', anchor_index_d)

anchor_nodes = [random_anchor_index, anchor_index_b, anchor_index_c, anchor_index_d]
if split_subG_num == 3:
    anchor_nodes = [random_anchor_index, anchor_index_b, anchor_index_c]
print('anchor_nodes', anchor_nodes)     # anchor_nodes [9, 425, 4457]

# --------------------------------------------------------------------------------------------------


# ---------------------------  2. extend to k-hop nodes based on anchors -------------------------------------
subG_all_seed_nodes_dict = {}
outlier_split_dict = split_all_rare_categories(abnormal_classes_list, org_labels, split_subG_num, seeds)
for k in range(split_subG_num):
    sub_G_outliers = outlier_split_dict[str(k)]
    anchor_node_k = anchor_nodes[k]
    all_path_nodes = []
    subG_k_fail_case_count = 0
    for outlier_j in sub_G_outliers:
        try:
            shortest_path_k = nx.shortest_path(nx_largest_subG, source=anchor_node_k, target=outlier_j)
            all_path_nodes.extend(shortest_path_k)
        except Exception as e:
            subG_k_fail_case_count += 1
            continue
    sub_G_seed_nodes_list = list(set(all_path_nodes))
    subG_all_seed_nodes_dict[str(k)] = sub_G_seed_nodes_list
for k in range(split_subG_num):
    sub_G_seed_nodes_all = subG_all_seed_nodes_dict[str(k)]
    extension_noides_num = int(len(sub_G_seed_nodes_all) * extension_ratio)
    sub_G_nodes_extended = []
    for node_k in sub_G_seed_nodes_all[0: extension_noides_num]:
        all_neighbors_1_hop, _, all_neighbors_3_hop, _ = find_all_1234_hop_Neighbor(nx_largest_subG, node_k, self_loop=False)
        sub_G_nodes_extended.extend(all_neighbors_3_hop)
    sub_G_nodes_extended = list(set(sub_G_nodes_extended))
    subG_all_seed_nodes_dict[str(k)].extend(sub_G_nodes_extended)

subG_all_nodes_drop_outliers_dict = {}
for j in range(split_subG_num):
    subG_j_nodes = subG_all_seed_nodes_dict[str(j)]
    subG_j_labels = gnd_binary[np.array(subG_j_nodes)]
    subG_j_outliers_local_index = np.where(subG_j_labels == 1)[0]
    subG_j_outliers_index = np.array(subG_j_nodes)[subG_j_outliers_local_index]
    sub_G_outliers_indx_list = outlier_split_dict[str(j)].tolist()
    non_overlap_outliers = list( set(subG_j_outliers_index.tolist()).intersection(set(sub_G_outliers_indx_list)) )
    for outlier_i in subG_j_outliers_index:
        if outlier_i not in sub_G_outliers_indx_list:
            subG_j_nodes.remove(outlier_i)
    subG_all_nodes_drop_outliers_dict[str(j)] = subG_j_nodes
# --------------------------------------------------------------------------------------------------


# ---------------------------  3. split the sub_graph -------------------------------------
DGL_Graph = dgl.from_scipy(adj_csr_matrix)
gnd_binary_outliers = np.where(gnd_binary == 1)[0]
for j in range(split_subG_num):
    sub_graph_nodes_list = subG_all_nodes_drop_outliers_dict[str(j)]
    new_adj_sparse, new_attribute_mat, new_labels, org_node_indices = split_graph(DGL_Graph, sub_graph_nodes_list, attri_matrix, gnd_binary)
    # fixed anomalies
    subG_outliers_new_indx = np.where(new_labels == 1)[0]
    fixed_20_anomaly_subG_k = shuffle_array(subG_outliers_new_indx, seed=seeds+999)[0:20]
    # -------------------------------------------
    final_mat_file = save_data_path + data_source + '_sub' + str(j)+'.mat'
    scipy.io.savemat(final_mat_file, {'Network': new_adj_sparse, 'Attributes': new_attribute_mat, 'gnd': new_labels})
    np.savetxt(save_data_path + data_source + '_sub' + str(j) + '_labeled_20anomaly' + '.csv', fixed_20_anomaly_subG_k, delimiter=',', fmt='%d')
    print('finished', str(j), 'th   subgraph ', '\n\n\n\n ')
