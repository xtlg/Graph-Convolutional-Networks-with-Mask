import csv
import pandas as pd
import os
from utils import util


def result_analysis(data_file_path, config):
    test_list = os.listdir(data_file_path)
    err_list = []
    pos_list = []
    for i in range(len(test_list)):

        if '_pos' in test_list[i]:
            pos_list.append(test_list[i])
        elif'err_y' in test_list[i]:
            err_list.append(test_list[i])
    for i in range(len(err_list)):

        err_path = data_file_path + '/' + err_list[i]
        pos_path = data_file_path + '/' + pos_list[i]
        err_data = pd.read_csv(err_path, header=None).to_numpy()
        pos_data = pd.read_csv(pos_path, header=None).to_numpy()

        for m in range(config['n_features']):
            nodes_err_info = []
            for j in range(config['n_nodes']):
                node_err_info = []
                for k in range(err_data.shape[0]):
                    if pos_data[k, j + m * config['n_nodes']] != 0:
                        node_err_info.append(err_data[k, j + m * config['n_nodes']])
                nodes_err_info.append(node_err_info)
            file_path = data_file_path + '/processed/' + str(pos_list[i]) + 'f' + str(m+1) + '.csv'
            with open(file_path, mode='w', newline='') as f:
                csv_writer = csv.writer(f)
                for row in range(len(nodes_err_info)):
                    csv_writer.writerow(nodes_err_info[row])

    return

config_path='../config/config_mask.yaml'
config = util.load_config(config_path)
result_analysis('D:/Education/Projects/开题/数据补全/newcodev5/results/test/MLP', config)
result_analysis('D:/Education/Projects/开题/数据补全/newcodev5/results/test/GCN', config)
result_analysis('D:/Education/Projects/开题/数据补全/newcodev5/results/test/GCNM', config)
result_analysis('D:/Education/Projects/开题/数据补全/newcodev5/results/test/GAN', config)
result_analysis('D:/Education/Projects/开题/数据补全/newcodev5/results/test/MLP_M', config)