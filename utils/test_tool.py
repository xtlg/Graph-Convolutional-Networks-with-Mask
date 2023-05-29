import numpy as np
import torch
from torch import nn
from utils import util
from models import GCN, MLP, Generator, MLP_M, MLP_M5, GCN_1, GCN_Full
from sklearn.preprocessing import MinMaxScaler
import os
from utils.util import Dataset


def GCN_test(model_file_path, data_path, test_file_path, adj_path, config_path,
             test_result_path='./results/test/GCN'):
    test_list = os.listdir(test_file_path)

    config = util.load_config(config_path)
    dataset = util.load_data(data_path)

    test_set = dataset[:config['test_len'], : config['n_features'] * config['n_nodes']]
    test_scaler = MinMaxScaler()
    dataset = test_scaler.fit_transform(test_set)
    adj = util.get_adj(adj_path).cuda()

    GCN_net = GCN(config)
    GCN_load = torch.load(model_file_path)
    GCN_net.load_state_dict(GCN_load['state_dict'])
    GCN_net.cuda()

    criterion = nn.MSELoss()
    loss_list = []
    for i in test_list:
        pos_path = test_file_path + '/' + i
        pos_set = util.load_data(pos_path)
        test_set = Dataset(dataset[:, ], pos_set[:config['test_len'], : config['n_features'] * config['n_nodes']],
                           config)
        x = torch.FloatTensor(test_set.features).cuda()
        yr = torch.FloatTensor(test_set.targets).cuda()
        z = torch.FloatTensor(-1 * (test_set.pos - 1)).cuda()
        y = GCN_net(x, z, adj)
        loss = criterion(y * z, yr * z)

        test_y = y.cpu().detach().numpy()
        true_y = yr.cpu().detach().numpy()
        pos = z.cpu().detach().numpy()

        inversed_test_y = test_scaler.inverse_transform(test_y)
        inversed_true_y = test_scaler.inverse_transform(true_y)

        f_name1 = test_result_path + '/' + i[6] + i[-6:-4] + str(config['mask']) + '_err_y.csv'
        f_name2 = test_result_path + '/' + i[6] + i[-6:-4] + str(config['mask']) + '_pos.csv'
        f_name3 = test_result_path + '/inversed/' + i[6] + i[-6:-4] + str(config['mask']) + '_inversed_test_y.csv'
        f_name4 = test_result_path + '/inversed/' + i[6] + i[-6:-4] + str(config['mask']) + '_inversed_true_y.csv'

        np.savetxt(f_name1, test_y - true_y, delimiter=',')
        np.savetxt(f_name2, pos, delimiter=',')
        np.savetxt(f_name3, inversed_test_y, delimiter=',')
        np.savetxt(f_name4, inversed_true_y, delimiter=',')

        loss_list.append(loss.item())
    f_name = test_result_path + '/GCN_mask_' + str(config['mask']) + 'test_loss_set.csv'
    np.savetxt(f_name, np.array(loss_list))


def MLP_test(model_file_path, data_path, test_file_path, config_path,
             test_result_path='./results/test/MLP'):
    test_list = os.listdir(test_file_path)

    config = util.load_config(config_path)
    dataset = util.load_data(data_path)

    test_set = dataset[:config['test_len'], : config['n_features'] * config['n_nodes']]
    test_scaler = MinMaxScaler()
    dataset = test_scaler.fit_transform(test_set)

    MLP_net = MLP(config)
    MLP_load = torch.load(model_file_path)
    MLP_net.load_state_dict(MLP_load['state_dict'])
    MLP_net.cuda()

    criterion = nn.MSELoss()
    loss_list = []
    for i in test_list:
        pos_path = test_file_path + '/' + i
        pos_set = util.load_data(pos_path)
        test_set = Dataset(dataset[:, ], pos_set[:config['test_len'], :config['n_features'] * config['n_nodes']],
                           config)
        x = torch.FloatTensor(test_set.features).cuda()
        yr = torch.FloatTensor(test_set.targets).cuda()
        z = torch.FloatTensor(-1 * (test_set.pos - 1)).cuda()
        y = MLP_net(x)
        loss = criterion(y * z, yr * z)

        test_y = y.cpu().detach().numpy()
        true_y = yr.cpu().detach().numpy()
        pos = z.cpu().detach().numpy()

        inversed_test_y = test_scaler.inverse_transform(test_y)
        inversed_true_y = test_scaler.inverse_transform(true_y)

        f_name1 = test_result_path + '/' + i[6] + i[-6:-4] + '_MLP_err_y.csv'
        f_name2 = test_result_path + '/' + i[6] + i[-6:-4] + '_MLP_pos.csv'
        f_name3 = test_result_path + '/inversed/' + i[6] + i[-6:-4] + str(config['mask']) + '_inversed_test_y.csv'
        f_name4 = test_result_path + '/inversed/' + i[6] + i[-6:-4] + str(config['mask']) + '_inversed_true_y.csv'
        np.savetxt(f_name1, test_y - true_y, delimiter=',')
        np.savetxt(f_name2, pos, delimiter=',')
        np.savetxt(f_name3, inversed_test_y, delimiter=',')
        np.savetxt(f_name4, inversed_true_y, delimiter=',')

        loss_list.append(loss.item())
    f_name = test_result_path + '/MLP_test_loss_set.csv'
    np.savetxt(f_name, np.array(loss_list))


def generator_test(model_file_path, data_path, test_file_path, config_path,
                   test_result_path='./results/test/GAN'):
    test_list = os.listdir(test_file_path)

    config = util.load_config(config_path)
    dataset = util.load_data(data_path)

    test_set = dataset[:config['test_len'], : config['n_features'] * config['n_nodes']]
    test_scaler = MinMaxScaler()
    dataset = test_scaler.fit_transform(test_set)

    generator = Generator(config)
    generator_load = torch.load(model_file_path)
    generator.load_state_dict(generator_load['state_dict'])
    generator.cuda()

    criterion = nn.MSELoss()
    loss_list = []
    for i in test_list:
        pos_path = test_file_path + '/' + i
        pos_set = util.load_data(pos_path)
        test_set = Dataset(dataset[:, ], pos_set[:config['test_len'], : config['n_features'] * config['n_nodes']],
                           config)
        x = torch.FloatTensor(test_set.features).cuda()
        yr = torch.FloatTensor(test_set.targets).cuda()
        z = torch.FloatTensor(-1 * (test_set.pos - 1)).cuda()
        y = generator(x)
        loss = criterion(y * z, yr * z)

        test_y = y.cpu().detach().numpy()
        true_y = yr.cpu().detach().numpy()
        pos = z.cpu().detach().numpy()
        inversed_test_y = test_scaler.inverse_transform(test_y)
        inversed_true_y = test_scaler.inverse_transform(true_y)

        f_name1 = test_result_path + '/' + i[6] + i[-6:-4] + '_GAN_err_y.csv'
        f_name2 = test_result_path + '/' + i[6] + i[-6:-4] + '_GAN_pos.csv'
        f_name3 = test_result_path + '/inversed/' + i[6] + i[-6:-4] + '_inversed_test_y.csv'
        f_name4 = test_result_path + '/inversed/' + i[6] + i[-6:-4] + '_inversed_true_y.csv'
        np.savetxt(f_name1, test_y - true_y, delimiter=',')
        np.savetxt(f_name2, pos, delimiter=',')
        np.savetxt(f_name3, inversed_test_y, delimiter=',')
        np.savetxt(f_name4, inversed_true_y, delimiter=',')

        loss_list.append(loss.item())
    f_name = test_result_path + '/GAN_test_loss_set.csv'
    np.savetxt(f_name, np.array(loss_list))


def MLP_M_test(model_file_path, data_path, test_file_path, config_path,
               test_result_path='./results/test/MLP_M'):
    test_list = os.listdir(test_file_path)

    config = util.load_config(config_path)
    dataset = util.load_data(data_path)

    test_set = dataset[:config['test_len'], : config['n_features'] * config['n_nodes']]
    test_scaler = MinMaxScaler()
    dataset = test_scaler.fit_transform(test_set)

    MLP_M_net = MLP_M(config)
    MLP_M_load = torch.load(model_file_path)
    MLP_M_net.load_state_dict(MLP_M_load['state_dict'])
    MLP_M_net.cuda()

    criterion = nn.MSELoss()
    loss_list = []
    for i in test_list:
        pos_path = test_file_path + '/' + i
        pos_set = util.load_data(pos_path)
        test_set = Dataset(dataset[:, ], pos_set[:config['test_len'], :config['n_features'] * config['n_nodes']],
                           config)
        x = torch.FloatTensor(test_set.features).cuda()
        yr = torch.FloatTensor(test_set.targets).cuda()
        z = torch.FloatTensor(-1 * (test_set.pos - 1)).cuda()
        y = MLP_M_net(x, z)
        loss = criterion(y * z, yr * z)

        test_y = y.cpu().detach().numpy()
        true_y = yr.cpu().detach().numpy()
        pos = z.cpu().detach().numpy()

        inversed_test_y = test_scaler.inverse_transform(test_y)
        inversed_true_y = test_scaler.inverse_transform(true_y)

        f_name1 = test_result_path + '/' + i[6] + i[-6:-4] + '_MLP_M_err_y.csv'
        f_name2 = test_result_path + '/' + i[6] + i[-6:-4] + '_MLP_M_pos.csv'
        f_name3 = test_result_path + '/inversed/' + i[6] + i[-6:-4] + str(config['mask']) + '_inversed_test_y.csv'
        f_name4 = test_result_path + '/inversed/' + i[6] + i[-6:-4] + str(config['mask']) + '_inversed_true_y.csv'
        np.savetxt(f_name1, test_y - true_y, delimiter=',')
        np.savetxt(f_name2, pos, delimiter=',')
        np.savetxt(f_name3, inversed_test_y, delimiter=',')
        np.savetxt(f_name4, inversed_true_y, delimiter=',')

        loss_list.append(loss.item())
    f_name = test_result_path + '/MLP_M_test_loss_set.csv'
    np.savetxt(f_name, np.array(loss_list))

    
def GCN_test_under_noise(model_file_path, data_path, test_file_path, adj_path, config_path,
             test_result_path='./results/test/GCN'):
    test_list = os.listdir(test_file_path)

    config = util.load_config(config_path)
    dataset = util.load_data(data_path)

    test_set = dataset[:config['test_len'], : config['n_features'] * config['n_nodes']]
    test_scaler = MinMaxScaler()
    dataset = test_scaler.fit_transform(test_set)

    adj = util.get_adj(adj_path).cuda()

    GCN_net = GCN(config)
    GCN_load = torch.load(model_file_path)
    GCN_net.load_state_dict(GCN_load['state_dict'])
    GCN_net.cuda()

    criterion = nn.MSELoss()
    loss_list = []
    for i in test_list:
        pos_path = test_file_path + '/' + i
        pos_set = util.load_data(pos_path)
        test_set = Dataset(dataset[:, ], pos_set[:config['test_len'], : config['n_features'] * config['n_nodes']],
                           config)

        x = torch.FloatTensor(test_set.features).cuda()
        # add noise
        noise = torch.normal(0, 0.1, (x.shape[0], x.shape[1])).cuda()
        x = x + noise

        yr = torch.FloatTensor(test_set.targets).cuda()
        z = torch.FloatTensor(-1 * (test_set.pos - 1)).cuda()
        y = GCN_net(x, z, adj)
        loss = criterion(y * z, yr * z)

        test_y = y.cpu().detach().numpy()
        true_y = yr.cpu().detach().numpy()
        pos = z.cpu().detach().numpy()

        # inversed_test_y = test_scaler.inverse_transform(test_y)
        # inversed_true_y = test_scaler.inverse_transform(true_y)
        #
        # f_name1 = test_result_path + '/' + i[6] + i[-6:-4] + str(config['mask']) + '_err_y.csv'
        # f_name2 = test_result_path + '/' + i[6] + i[-6:-4] + str(config['mask']) + '_pos.csv'
        # f_name3 = test_result_path + '/inversed/' + i[6] + i[-6:-4] + str(config['mask']) + '_inversed_test_y.csv'
        # f_name4 = test_result_path + '/inversed/' + i[6] + i[-6:-4] + str(config['mask']) + '_inversed_true_y.csv'
        #
        # np.savetxt(f_name1, test_y - true_y, delimiter=',')
        # np.savetxt(f_name2, pos, delimiter=',')
        # np.savetxt(f_name3, inversed_test_y, delimiter=',')
        # np.savetxt(f_name4, inversed_true_y, delimiter=',')

        loss_list.append(loss.item())
    f_name = test_result_path + '/GCN_mask_' + str(config['mask']) + 'test_loss_set.csv'
    np.savetxt(f_name, np.array(loss_list))
