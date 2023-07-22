import numpy as np
import scipy.sparse as sp
import torch
import sys
from sklearn.metrics import f1_score
import scipy
import scipy.io
from sklearn.model_selection import train_test_split
import copy
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import random


def shuffle_array(input_arr, seed):
    total_node = input_arr.shape[0]
    randomlist = [i for i in range(total_node)]
    random.seed(seed)
    random.shuffle(randomlist)
    ouptut_arr = input_arr[randomlist]
    return ouptut_arr


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sgc_precompute(features, adj, degree):
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def load_subGraph_data(root_path, data_source):
    data = scipy.io.loadmat(root_path+"{}.mat".format(data_source))
    gnds = data["gnd"]
    attributes_sprse = sp.csr_matrix(data["Attributes"])
    network = sp.lil_matrix(data["Network"])
    adj_norm = preprocess_adj(network)
    attributes = attributes_sprse.todense()
    reshaped_gnd = gnds.reshape(attributes.shape[0], -1)
    return adj_norm, attributes, reshaped_gnd


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def GAD_train_test_split(labels, random_seed, known_outliers_num, train_ratio, val_ratio, args, j):
    modified_gnd = copy.deepcopy(labels)
    total_nodes_num = labels.shape[0]
    outliers_index = np.where(labels == 1)[0]
    outliers_num = outliers_index.shape[0]
    normal_index = np.where(labels == 0)[0]
    labeled_trn_outlier_num = int(known_outliers_num)
    trn_val_norm_indx, test_norm_indx = train_test_split(normal_index, test_size=1 - train_ratio - val_ratio, random_state=random_seed + 23)

    subG_j_labeled_outliers = np.loadtxt(args.data_path + args.dataset + str(j)+'_labeled_20anomaly.csv', delimiter=',', dtype=int)
    test_abnrm_num = int(outliers_num * (1 - train_ratio - val_ratio))
    exclude_trn_outlier = np.array(list(set(outliers_index).difference(set(subG_j_labeled_outliers))))
    shuffled_left_outliers = shuffle_array(exclude_trn_outlier, seed=random_seed + 3)
    test_abnrm_indx = shuffled_left_outliers[0:test_abnrm_num]
    trn_val_masked_outlier_indx = shuffled_left_outliers[test_abnrm_num:]
    train_all_outlier_set = subG_j_labeled_outliers[0:labeled_trn_outlier_num]
    trn_val_abnrm_indx = np.append(train_all_outlier_set, trn_val_masked_outlier_indx, axis=0)
    for indx_k in trn_val_masked_outlier_indx:
        modified_gnd[indx_k] = 0
    current_outlier_indices = np.where(modified_gnd == 1)[0]
    outliers_as_normal_in_trn_val = list(set(outliers_index).difference(set(current_outlier_indices)))
    train_indx_final = np.append(trn_val_norm_indx, trn_val_abnrm_indx, axis=0)
    test_index_final = np.append(test_norm_indx, test_abnrm_indx, axis=0)
    train_all_normal_set = np.append(trn_val_norm_indx, trn_val_masked_outlier_indx,axis=0)
    return train_all_normal_set, train_all_outlier_set, test_index_final, modified_gnd


def train_batch_iter(outlier_indices, inlier_indices, batch_size, seed):
    train_indx_list_norm = []
    train_indx_list_abnorm = []
    sampled_inliers_num = int(0.5 * batch_size)
    assert sampled_inliers_num <= inlier_indices.shape[0]
    n_inliers = inlier_indices.shape[0]
    n_outliers = outlier_indices.shape[0]
    inlier_indices = shuffle_array(inlier_indices, seed)
    train_indx_list_norm.extend(list(inlier_indices[0:sampled_inliers_num]))
    over_sample_factor = int(sampled_inliers_num / n_outliers) + 1
    outlier_indices_repeat = np.tile(outlier_indices, over_sample_factor)
    outlier_indices_repeat = shuffle_array(outlier_indices_repeat, seed)
    train_indx_list_abnorm.extend(list(outlier_indices_repeat[0:sampled_inliers_num]))

    return np.array(train_indx_list_abnorm), np.array(train_indx_list_norm)


def generate_support_query_Meta(totalTask_Num, SubGraph_info, trn_batch_size, args, current_epoch):
    x_spt, y_spt = [], []
    attri_dim = SubGraph_info["features_" + str(0)].shape[1]
    device = SubGraph_info["features_" + str(0)].device
    Metatask_num = 20
    remain_prop = args.remain_prop
    for j in range(Metatask_num):
        seed_changing = args.seed + current_epoch**2 + j**2
        SumUp_spt_x = torch.from_numpy(np.zeros((1, attri_dim), dtype=np.float32)).to(device)
        SumUp_spt_y = torch.from_numpy(np.zeros((1, 1))).to(device)
        for t in range(totalTask_Num):
            seed1 = seed_changing + t + 5678
            features_Gi = SubGraph_info["features_" + str(t)]
            label_Gi = SubGraph_info['all_label_' + str(t)]
            org_labeled_anom_Gi = SubGraph_info['trn_outlier_indx_' + str(t)]
            seeds_changing = seed1 + 1024
            shuff_masked_trn_outlier = shuffle_array(org_labeled_anom_Gi, seeds_changing)
            remain_outlier_num = int(shuff_masked_trn_outlier.shape[0])
            remain_trn_outliers_Gi = shuff_masked_trn_outlier[0:remain_outlier_num]

            org_trn_norm = SubGraph_info['trn_norm_indx_' + str(t)]
            shuff_masked_trn_norm = shuffle_array(org_trn_norm, seed1)
            remain_num = int(org_trn_norm.shape[0] * remain_prop)
            remain_masked_trn_norm = shuff_masked_trn_norm[0:remain_num]
            over_sample_factor = int(1 / remain_prop) + 1
            extend_masked_trn_norm = np.tile(remain_masked_trn_norm, over_sample_factor)
            final_masked_trn_norm = extend_masked_trn_norm[0:org_trn_norm.shape[0]]
            spt_outlier_ind, spt_norm_ind = train_batch_iter(remain_trn_outliers_Gi, final_masked_trn_norm, trn_batch_size,  seed=seed1+12)
            tmp_merge_x = torch.cat((features_Gi[spt_outlier_ind], features_Gi[spt_norm_ind]), 0)
            tmp_merge_y = torch.cat((label_Gi[spt_outlier_ind], label_Gi[spt_norm_ind]), 0)
            SumUp_spt_x = torch.cat((SumUp_spt_x, tmp_merge_x), 0)
            SumUp_spt_y = torch.cat((SumUp_spt_y, tmp_merge_y), 0)
        final_spt_x = SumUp_spt_x[1:, :]
        final_spt_y = SumUp_spt_y[1:, :]
        x_spt.append(final_spt_x)
        y_spt.append(final_spt_y)
    return x_spt, y_spt


def evaluation_metric(pred_anomaly_score, gnds):
    roc_auc = roc_auc_score(gnds, pred_anomaly_score)
    roc_pr_area = average_precision_score(gnds, pred_anomaly_score)
    all_results = []
    all_results.append(roc_auc)
    all_results.append(roc_pr_area)
    return all_results


def load_checkpoints(model, checkpoint_path, mode='Test'):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if mode == 'Train':
        model.train()
    if mode == 'Test':
        model.eval()
    return model


def select_similar_nodes(source_node_emb, node_list, attri_mat, top_k_node):
    def Euclidien_distance(vec1, vec2):
        Euclidien_score = np.linalg.norm(vec1 - vec2)
        return Euclidien_score
    Euclidien_distance_list = []
    for j in node_list:
        similarity_score_j = Euclidien_distance(source_node_emb, attri_mat[j])
        Euclidien_distance_list.append(similarity_score_j)
    sorted_index = np.argsort(np.array(Euclidien_distance_list))
    longest_distance = Euclidien_distance_list[top_k_node+1]
    return list(np.array(node_list)[sorted_index])[1:top_k_node+1], longest_distance


def external_diversify(anom_emb_k, current_G_indx, Emb_dict, args, distance_threshold, auged_anom_dict, similar_nodes_num):
    for ex_j in range(args.totalTask_num):
        if ex_j != current_G_indx:
            ex_G_nodes = Emb_dict[''+str(ex_j)].shape[0]
            ex_G_emb = Emb_dict[''+str(ex_j)]
            def Euclidien_distance(vec1, vec2):
                Euclidien_score = np.linalg.norm(vec1 - vec2)
                return Euclidien_score
            Euclidien_distance_list = []
            for j in range(ex_G_nodes):
                similarity_score_j = Euclidien_distance(anom_emb_k, ex_G_emb[j])
                Euclidien_distance_list.append(similarity_score_j)
            Euclidien_distance_arr = np.array(Euclidien_distance_list)
            sorted_index = np.argsort(Euclidien_distance_arr)
            sorted_distace = Euclidien_distance_arr[sorted_index]
            if sorted_distace[0] >= distance_threshold:
                pass
            else:
                satisfied_Gj_id = np.where(Euclidien_distance_arr < distance_threshold)[0]
                satisfied_nodes_distan = Euclidien_distance_arr[satisfied_Gj_id]
                sorted_index2 = np.argsort(satisfied_nodes_distan)
                most_closet_nodes = satisfied_Gj_id[sorted_index2][0:similar_nodes_num]
                auged_anom_dict[''+str(ex_j)].extend(list(most_closet_nodes))
    return auged_anom_dict





