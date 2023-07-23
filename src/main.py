import torch
import numpy as np
import argparse

from utils import sgc_precompute, set_seed, load_subGraph_data, \
    GAD_train_test_split, sparse_mx_to_torch_sparse_tensor, \
    generate_support_query_Meta, evaluation_metric, load_checkpoints,\
    select_similar_nodes, external_diversify
from meta import Meta_episodic
from mlp import MLP
import pandas as pd
import os


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda:1')
    if args.dataset in ['CS_sub', 'Amazon_clothing_sub', 'DBLP_sub']:
        args.totalTask_num = 4 - 1
    elif args.dataset in ['Photo_sub', 'ACM_sub']:
        args.totalTask_num = 3 - 1  # leave test

    SubGraph_info = {}
    for k in range(args.totalTask_num+1):
        adj_norm_k, features_k, gnds_k = 'adj_norm_' + str(k), "features_" + str(k), "gnds_" + str(k)
        trn_norm_indx_k, trn_outlier_indx_k, test_indx_final_k, y_semi_k = 'trn_norm_indx_' + str(k), 'trn_outlier_indx_' + str(k), \
                                                                       'test_indx_final_' + str(k), 'y_semi_' + str(k)
        all_label_k, adj_k = 'all_label_' + str(k), 'adj_' + str(k)
        SubGraph_info[adj_norm_k], SubGraph_info[features_k], SubGraph_info[gnds_k] = load_subGraph_data(args.data_path, args.dataset + str(k))
        subG_nodes_num = SubGraph_info[features_k].shape[0]
        if k != args.totalTask_num:
            SubGraph_info[trn_norm_indx_k], SubGraph_info[trn_outlier_indx_k], SubGraph_info[test_indx_final_k], SubGraph_info[y_semi_k] = \
                GAD_train_test_split(SubGraph_info[gnds_k], args.seed + k, args.known_outliers_num, args.trn_ratio, args.val_ratio, args, k)
            SubGraph_info[all_label_k] = torch.from_numpy(SubGraph_info[y_semi_k]).to(device)
        else:
            SubGraph_info[test_indx_final_k] = np.array(range(subG_nodes_num))
        SubGraph_info[adj_k] = sparse_mx_to_torch_sparse_tensor(SubGraph_info[adj_norm_k]).to(device)
        SubGraph_info[features_k] = torch.FloatTensor(np.array(SubGraph_info[features_k])).to(device)
        SubGraph_info[features_k] = sgc_precompute(SubGraph_info[features_k], SubGraph_info[adj_k], args.degree)
    attribute_dim = SubGraph_info["features_" + str(0)].size(1)
    config = [
        ('linear', [args.hidden1, attribute_dim]),
        ('linear', [1, args.hidden1])
    ]
    trn_batch_size = args.batch_size
    min_trn_loss = 1000
    wait_count = 0
    MLP_init = MLP(args, attribute_dim).to(device)
    Emb_dict = {}
    auged_anom_dict = {}
    for j in range(args.totalTask_num):
        feat_j = SubGraph_info['features_' + str(j)]
        emb_j, _ = MLP_init(feat_j)
        Emb_dict['' + str(j)] = emb_j.detach().cpu().numpy()
        auged_anom_dict['' + str(j)] = []
    for j in range(args.totalTask_num):
        emb_j = Emb_dict['' + str(j)]
        labeled_outliers = SubGraph_info['trn_outlier_indx_' + str(j)]
        Gj_nodes_num = emb_j.shape[0]
        Gj_node_index = np.array([i for i in range(Gj_nodes_num)])
        similar_nodes_num = 1
        for node_k in labeled_outliers:
            anom_emb_k = emb_j[node_k]
            _, longest_distance = select_similar_nodes(anom_emb_k, Gj_node_index, emb_j, similar_nodes_num)
            longest_distance = args.alpha * longest_distance
            auged_anom_dict = external_diversify(anom_emb_k, j, Emb_dict, args, longest_distance, auged_anom_dict, similar_nodes_num)
    for j in range(args.totalTask_num):
        labeled_outliers = SubGraph_info['trn_outlier_indx_' + str(j)]
        trn_normal = SubGraph_info['trn_norm_indx_' + str(j)]
        org_outlier_in_normal = list(set(labeled_outliers).intersection(set(trn_normal)))
        assert len(org_outlier_in_normal) == 0
        augmented_outlier = auged_anom_dict['' + str(j)]
        rough_total_outlier = np.append(labeled_outliers, np.array(augmented_outlier), axis=0)
        total_auged_outlier = np.array(list(set(rough_total_outlier)))
        SubGraph_info['trn_outlier_indx_' + str(j)] = total_auged_outlier
        auged_in_normal = list(set(total_auged_outlier).intersection(set(trn_normal)))
        gnd_outliers = np.where(SubGraph_info["gnds_" + str(j)] == 1)[0]
        auged_is_outlier = list(set(total_auged_outlier).intersection(set(gnd_outliers)))
        purified_normal = list(set(trn_normal).difference(set(total_auged_outlier)))
        SubGraph_info['trn_norm_indx_' + str(j)] = np.array(purified_normal)

    for i in range(1):
        maml = Meta_episodic(args, config).to(device)
        trn_loss_list = []
        for j in range(args.epoch):
            x_spt, y_spt = generate_support_query_Meta(args.totalTask_num, SubGraph_info, trn_batch_size, args, j)
            x_qry, y_qry = generate_support_query_Meta(args.totalTask_num, SubGraph_info, trn_batch_size, args, j+7)
            trn_loss = maml(x_spt, y_spt, x_qry, y_qry)
            print('Epoch:', j, '\tMeta_Training_Loss: {:.6f}'.format(trn_loss))
            trn_loss_list.append(trn_loss)
            current_loss = trn_loss
            if current_loss < min_trn_loss:
                wait_count = 0
                min_trn_loss = current_loss
                checkpoint = {'model': maml, 'model_state_dict': maml.state_dict()}
                model_path_now = os.path.join(args.model_path, '{}.pkl'.format(args.dataset))
                torch.save(checkpoint, model_path_now)
            else:
                wait_count += 1
                if wait_count > args.patience and args.early_stop:
                    break
        model_path_now = os.path.join(args.model_path, '{}.pkl'.format(args.dataset))
        loaded_maml = load_checkpoints(maml, model_path_now, mode='Test')
        test_set_pred = []
        test_set_gnd = []
        for j in range(args.totalTask_num+1):
            test_feat_mat = SubGraph_info['features_' + str(j)]
            test_indx = SubGraph_info['test_indx_final_' + str(j)]
            test_Task_id = j
            final_pred_score = loaded_maml.testing_GAD(test_feat_mat)
            anomaly_score = final_pred_score.detach().cpu().numpy()
            test_y_pred_anomaly_score = anomaly_score[test_indx]
            y_true = np.array([label[0] for label in SubGraph_info['gnds_' + str(test_Task_id)]])
            test_gnd = y_true[test_indx]
            if j < args.totalTask_num:
                test_pred_score = [i[0] for i in test_y_pred_anomaly_score]
                test_set_pred.extend(test_pred_score)
                test_set_gnd.extend(test_gnd.tolist())
            if j == args.totalTask_num:
                unseen_G_results = evaluation_metric(test_y_pred_anomaly_score, test_gnd)
                df = pd.DataFrame(np.array(unseen_G_results).reshape(-1, 2), columns=['AUC', 'AUPR'])
                df.to_csv(args.path_results + 'results_Unseen_' + args.dataset + '_outliers_' + str(args.known_outliers_num)
                          + '_seed_' + str(args.seed) + '.csv')
        results_metric_AD = evaluation_metric(np.array(test_set_pred), np.array(test_set_gnd))
        df = pd.DataFrame(np.array(results_metric_AD).reshape(-1, 2), columns=['AUC', 'AUPR'])
        df.to_csv(args.path_results+'results_' + args.dataset + '_outliers_' + str(args.known_outliers_num)
             + '_seed_' + str(args.seed) + '.csv')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--epoch', type=int, help='epoch number', default=2001)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.0001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.0001)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--dataset', type=str, default='AD_amazon_electronics_photo_sub', help='Dataset')
    argparser.add_argument('--normalization', type=str, default='AugNormAdj', help='Normalization method for the adjacency matrix.')
    argparser.add_argument('--seed', type=int, default=42, help='Random seed.')
    argparser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
    argparser.add_argument('--hidden1', type=int, help='Number of hidden units', default=512)
    argparser.add_argument('--hidden2', type=int, help='Number of hidden units', default=128)
    argparser.add_argument('--known_outliers_num', type=int, default=20, help='the known anomalies')
    argparser.add_argument('--trn_ratio', type=float, default=0.4, help='train data ratio')
    argparser.add_argument('--val_ratio', type=float, default=0.2, help='valid data ratio')
    argparser.add_argument('--tes_ratio', type=float, default=0.4, help='test data ratio')
    argparser.add_argument('--totalTask_num', type=int, default=2, help='the total number of tasks')
    argparser.add_argument("--batch_size", type=int, default=128, help=" batch size")
    argparser.add_argument("--data_path", type=str, default="../../sub_G_datasets/")
    argparser.add_argument("--path_results", type=str, default="../results/")
    argparser.add_argument("--model_path", type=str, default="../models/")
    argparser.add_argument('--query_diversity', type=int, default=3, help='query: outlier-normal pair diversity')
    argparser.add_argument('--factor', type=int, default=2, help='factor')
    argparser.add_argument('--early_stop', type=bool, default=True, help='whether early stop')
    argparser.add_argument('--patience', type=int, default=500, help='early stop')
    argparser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    argparser.add_argument('--alpha', type=float, default=0.5, help='weight to control the threshold')
    argparser.add_argument('--remain_prop', type=float, default=0.8, help='1-mask ratio; remained for training')
    args = argparser.parse_args()

    main(args)
