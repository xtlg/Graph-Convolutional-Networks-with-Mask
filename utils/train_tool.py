import torch
import time
import numpy as np

import models
import utils.util as util
from tqdm import tqdm
from torch import optim, nn
from torchinfo import summary
from utils.util import Dataset
from models import GCN, MLP, Generator, Discriminator, MLP_M
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataloader import DataLoader


def GCN_train(data_path, pos_path, adj_path, config_path, model_result_path='./results/model',
              loss_result_path='./results/loss', time_result_path='./results/time'):
    # data_path is the data file path, and the data is a complete power flow dataset
    # pos_path is the pos file path, using pos to choose where the node data missing only containing 0 or 1
    # adj_path is the adjacent matrix of the transmission grid file path
    # config_path is the config file path, config set the parameter of model and training
    # xxx_result is the path to save the trained model and the training loss

    config = util.load_config(config_path)
    dataset = util.load_data(data_path)
    pos_set = util.load_data(pos_path)

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    train_set = dataset[:6000, : config['n_features'] * config['n_nodes']]
    val_set = dataset[6000: 8000, :config['n_features'] * config['n_nodes']]
    train_pos = pos_set[:6000, :config['n_features'] * config['n_nodes']]
    val_pos = pos_set[6000: 8000, :config['n_features'] * config['n_nodes']]
    train_set = Dataset(train_set, train_pos, config)
    val_set = Dataset(val_set, val_pos, config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False)
    adj = util.get_adj(adj_path).cuda()

    GCN_net = GCN(config)
    summary(GCN_net)
    GCN_net.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(GCN_net.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.1,
                                                     patience=200,
                                                     verbose=True)

    GCN_train_loss = []
    GCN_val_loss = []

    start_time = time.time()
    for epoch in tqdm(range(config['epochs'])):

        loss_temp = 0
        n_set = 0

        GCN_net.train()
        for i, (features, targets, pos) in enumerate(train_loader):
            x = features.cuda()
            yr = targets.cuda()
            z = (-1 * (pos - 1)).cuda()
            y = GCN_net(x, z, adj)
            loss = criterion(y * z, yr * z)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_temp += loss.item()
            n_set += 1
        temp1 = loss_temp / n_set
        GCN_train_loss.append(temp1)

        GCN_net.eval()
        with torch.no_grad():
            val_x = torch.FloatTensor(val_set.features).cuda()
            val_yr = torch.FloatTensor(val_set.targets).cuda()
            val_z = torch.FloatTensor(-1 * (val_set.pos - 1)).cuda()

            y = GCN_net(val_x, val_z, adj).detach()
            val_loss = criterion(y * val_z, val_yr * val_z)
        scheduler.step(val_loss)
        GCN_val_loss.append(val_loss.cpu().item())

        if (epoch + 1) % 100 == 0:
            print('Epoch[%d/%d], train_loss:%.8f, val_loss:%.8f'
                  % (epoch + 1, config['epochs'], temp1, val_loss))
    run_time = np.array(time.time() - start_time).reshape(-1, 1)

    model_file_name = model_result_path + '/GCN_mask_' + str(config['mask']) + '_net.pth.tar'
    torch.save({'state_dict': GCN_net.state_dict()}, model_file_name)
    train_loss_file_name = loss_result_path + '/GCN_mask_' + str(config['mask']) + '_train_loss.csv'
    np.savetxt(train_loss_file_name, GCN_train_loss)
    val_loss_file_name = loss_result_path + '/GCN_mask_' + str(config['mask']) + '_val_loss.csv'
    np.savetxt(val_loss_file_name, GCN_val_loss)
    time_file_name = time_result_path + '/GCN_mask_' + str(config['mask']) + '_time.csv'
    np.savetxt(time_file_name, run_time)


def MLP_train(data_path, pos_path, config_path, model_result_path='./results/model',
              loss_result_path='./results/loss', time_result_path='./results/time'):
    config = util.load_config(config_path)
    dataset = util.load_data(data_path)
    pos_set = util.load_data(pos_path)

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    train_set = dataset[:6000, : config['n_features'] * config['n_nodes']]
    val_set = dataset[6000: 8000, :config['n_features'] * config['n_nodes']]
    train_pos = pos_set[:6000, :config['n_features'] * config['n_nodes']]
    val_pos = pos_set[6000: 8000, :config['n_features'] * config['n_nodes']]
    train_set = Dataset(train_set, train_pos, config)
    val_set = Dataset(val_set, val_pos, config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False)

    MLP_net = MLP(config)
    summary(MLP_net)
    MLP_net.cuda()

    optimizer = optim.Adam(MLP_net.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                     patience=200,
                                                     verbose=True)
    criterion = nn.MSELoss()

    MLP_train_loss = []
    MLP_val_loss = []

    start_time = time.time()
    for epoch in tqdm(range(config['epochs'])):

        loss_temp = 0
        n_set = 0

        MLP_net.train()
        for i, (features, targets, pos) in enumerate(train_loader):
            x = features.cuda()
            yr = targets.cuda()
            z = (-1 * (pos - 1)).cuda()
            y = MLP_net(x)
            loss = criterion(y * z, yr * z)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_temp += loss.item()
            n_set += 1
        temp1 = loss_temp / n_set
        MLP_train_loss.append(temp1)

        MLP_net.eval()
        with torch.no_grad():
            val_x = torch.FloatTensor(val_set.features).cuda()
            val_yr = torch.FloatTensor(val_set.targets).cuda()
            val_z = torch.FloatTensor(-1 * (val_set.pos - 1)).cuda()

            y = MLP_net(val_x).detach()
            val_loss = criterion(y * val_z, val_yr * val_z)

        scheduler.step(val_loss)
        MLP_val_loss.append(val_loss.cpu().item())
        if (epoch + 1) % 100 == 0:
            print('Epoch[%d/%d], train_loss:%.8f, val_loss:%.8f'
                  % (epoch + 1, config['epochs'], temp1, MLP_val_loss[-1]))
    run_time = np.array(time.time() - start_time).reshape(-1, 1)

    model_file_name = model_result_path + '/MLP_net.pth.tar'
    torch.save({'state_dict': MLP_net.state_dict()}, model_file_name)
    train_loss_file_name = loss_result_path + '/MLP_train_loss.csv'
    np.savetxt(train_loss_file_name, MLP_train_loss)
    val_loss_file_name = loss_result_path + '/MLP_val_loss.csv'
    np.savetxt(val_loss_file_name, MLP_val_loss)
    time_file_name = time_result_path + '/MLP_time.csv'
    np.savetxt(time_file_name, run_time)


def WGAN_train(data_path, pos_path, config_path, model_result_path='./results/model',
               loss_result_path='./results/loss', time_result_path='./results/time'):
    config = util.load_config(config_path)
    dataset = util.load_data(data_path)
    pos_set = util.load_data(pos_path)

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    train_set = dataset[:6000, : config['n_features'] * config['n_nodes']]
    val_set = dataset[6000: 8000, :config['n_features'] * config['n_nodes']]
    train_pos = pos_set[:6000, :config['n_features'] * config['n_nodes']]
    val_pos = pos_set[6000: 8000, :config['n_features'] * config['n_nodes']]
    train_set = Dataset(train_set, train_pos, config)
    val_set = Dataset(val_set, val_pos, config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False)

    generator = Generator(config)
    discriminator = Discriminator(config)

    summary(generator)
    summary(discriminator)
    generator.cuda()
    discriminator.cuda()

    criterion = nn.MSELoss()
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['learning_rate'])
    optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate'])

    generator_train_loss_part1_list = []
    generator_train_loss_part2_list = []
    generator_val_loss_part1_list = []
    generator_val_loss_part2_list = []
    discriminator_train_loss_list = []
    discriminator_val_loss_list = []

    start_time = time.time()
    for epoch in tqdm(range(config['epochs'])):

        generator_loss_temp_part1 = 0
        generator_loss_temp_part2 = 0
        generator_n_set = 0
        discriminator_loss_r_temp = 0
        discriminator_loss_f_temp = 0
        discriminator_n_set = 0

        generator.train()
        discriminator.train()

        for _ in range(100):
            for i, (features, targets, pos) in enumerate(train_loader):
                x = features.cuda()
                yr = targets.cuda()

                pred_r = discriminator(yr)
                yf = generator(x)
                pred_f = discriminator(yf)
                loss_r = -pred_r.mean()
                loss_f = pred_f.mean()
                loss_D = loss_r + 3 * loss_f

                loss_D.backward()
                optimizer_D.step()
                optimizer_D.zero_grad()

                discriminator_loss_r_temp += loss_r.item()
                discriminator_loss_f_temp += loss_f.item()
                discriminator_n_set += 1

                if loss_f < 0.3:
                    break
            if loss_f < 0.45:
                break

        dis_temp_f = discriminator_loss_f_temp / discriminator_n_set

        discriminator_train_loss_list.append(dis_temp_f)

        for _ in range(5):
            for i, (features, targets, pos) in enumerate(train_loader):
                x = features.cuda()
                yr = targets.cuda()
                z = (-1 * (pos - 1)).cuda()

                yf = generator(x)
                predf = discriminator(yf)
                loss_discriminator = -predf.mean()
                recover_mse = criterion(yf * z, yr * z)
                loss_G = loss_discriminator + recover_mse
                loss_G.backward()
                optimizer_G.step()
                optimizer_G.zero_grad()
                # scheduler.step(loss_G)
                generator_loss_temp_part2 += loss_discriminator.item()
                generator_loss_temp_part1 += recover_mse.item()

                generator_n_set += 1
                if -1 * loss_discriminator > 0.75:
                    break
            if -1 * loss_discriminator > 0.52:
                break

        gen_loss_part_2 = generator_loss_temp_part2 / generator_n_set
        generator_train_loss_part2_list.append(gen_loss_part_2)
        gen_loss_part_1 = generator_loss_temp_part1 / generator_n_set
        generator_train_loss_part1_list.append(gen_loss_part_1)

        generator.eval()
        discriminator.eval()

        with torch.no_grad():
            val_x = torch.FloatTensor(val_set.features).cuda()
            val_yr = torch.FloatTensor(val_set.targets).cuda()
            val_z = torch.FloatTensor(-1 * (val_set.pos - 1)).cuda()
            generator_val_out = generator(val_x)
            pred_f = discriminator(generator_val_out)

            discriminator_val_loss = pred_f.cpu().mean()
            generator_val_recover_mse = criterion(generator_val_out * val_z, val_yr * val_z)
            generator_val_loss = pred_f.cpu().mean()

        generator_val_loss_part2_list.append(generator_val_loss.item())
        generator_val_loss_part1_list.append(generator_val_recover_mse.item())
        discriminator_val_loss_list.append(discriminator_val_loss.item())

        if (epoch + 1) % 20 == 0:
            print('Epoch[%d/%d], generator_train_loss:%.8f, discriminator_train_loss:%.8f, train_recover_mse:%.8f,'
                  ' generator_val_loss:%.8f, discriminator_val_loss:%.8f, val_recover_mse:%.8f'
                  % (epoch + 1, config['epochs'], gen_loss_part_2, dis_temp_f, gen_loss_part_1,
                     generator_val_loss.item(), discriminator_val_loss.item(), generator_val_recover_mse.item()))
    run_time = time.time() - start_time

    generator_file_name = model_result_path + '/W_generator_net.pth.tar'
    discriminator_file_name = model_result_path + '/W_discriminator_net.pth.tar'
    torch.save({'state_dict': generator.state_dict()}, generator_file_name)
    torch.save({'state_dict': generator.state_dict()}, discriminator_file_name)
    generator_train_loss_part2_file_name = loss_result_path + '/W_generator_train_loss_part2.csv'
    generator_train_loss_part1_file_name = loss_result_path + '/W_generator_train_loss_part1.csv'
    discriminator_train_loss_file_name = loss_result_path + '/W_discriminator_train_loss.csv'
    np.savetxt(generator_train_loss_part2_file_name, generator_train_loss_part2_list, delimiter=',')
    np.savetxt(generator_train_loss_part1_file_name, generator_train_loss_part1_list, delimiter=',')
    np.savetxt(discriminator_train_loss_file_name, discriminator_train_loss_list, delimiter=',')
    generator_val_loss_part2_file_name = loss_result_path + '/W_generator_val_loss_part2.csv'
    generator_val_loss_part1_file_name = loss_result_path + '/W_generator_val_loss_part1.csv'
    discriminator_val_loss_file_name = loss_result_path + '/W_discriminator_val_loss.csv'
    np.savetxt(generator_val_loss_part2_file_name, generator_val_loss_part2_list, delimiter=',')
    np.savetxt(generator_val_loss_part1_file_name, generator_val_loss_part1_list, delimiter=',')
    np.savetxt(discriminator_val_loss_file_name, discriminator_val_loss_list, delimiter=',')
    time_file_name = time_result_path + '/WGAN_time.csv'
    np.savetxt(time_file_name, np.array(run_time).reshape(-1, 1), delimiter=',')


def MLP_M_train(data_path, pos_path, config_path, model_result_path='./results/model',
                loss_result_path='./results/loss', time_result_path='./results/time'):
    config = util.load_config(config_path)
    dataset = util.load_data(data_path)
    pos_set = util.load_data(pos_path)

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    train_set = dataset[:6000, : config['n_features'] * config['n_nodes']]
    val_set = dataset[6000: 8000, :config['n_features'] * config['n_nodes']]
    train_pos = pos_set[:6000, :config['n_features'] * config['n_nodes']]
    val_pos = pos_set[6000: 8000, :config['n_features'] * config['n_nodes']]
    train_set = Dataset(train_set, train_pos, config)
    val_set = Dataset(val_set, val_pos, config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False)

    MLP_M_net = MLP_M(config)
    summary(MLP_M_net)
    MLP_M_net.cuda()

    optimizer = optim.Adam(MLP_M_net.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                     patience=200,
                                                     verbose=True)
    criterion = nn.MSELoss()

    MLP_train_loss = []
    MLP_val_loss = []

    start_time = time.time()
    for epoch in tqdm(range(config['epochs'])):

        loss_temp = 0
        n_set = 0

        MLP_M_net.train()
        for i, (features, targets, pos) in enumerate(train_loader):
            x = features.cuda()
            yr = targets.cuda()
            z = (-1 * (pos - 1)).cuda()
            y = MLP_M_net(x, z)
            loss = criterion(y * z, yr * z)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_temp += loss.item()
            n_set += 1
        temp1 = loss_temp / n_set
        MLP_train_loss.append(temp1)

        MLP_M_net.eval()
        with torch.no_grad():
            val_x = torch.FloatTensor(val_set.features).cuda()
            val_yr = torch.FloatTensor(val_set.targets).cuda()
            val_z = torch.FloatTensor(-1 * (val_set.pos - 1)).cuda()

            y = MLP_M_net(val_x, val_z).detach()
            val_loss = criterion(y * val_z, val_yr * val_z)

        scheduler.step(val_loss)
        MLP_val_loss.append(val_loss.cpu().item())
        if (epoch + 1) % 100 == 0:
            print('Epoch[%d/%d], train_loss:%.8f, val_loss:%.8f'
                  % (epoch + 1, config['epochs'], temp1, MLP_val_loss[-1]))
    run_time = np.array(time.time() - start_time).reshape(-1, 1)

    model_file_name = model_result_path + '/MLP_M_net.pth.tar'
    torch.save({'state_dict': MLP_M_net.state_dict()}, model_file_name)
    train_loss_file_name = loss_result_path + '/MLP_M_train_loss.csv'
    np.savetxt(train_loss_file_name, MLP_train_loss)
    val_loss_file_name = loss_result_path + '/MLP_M_val_loss.csv'
    np.savetxt(val_loss_file_name, MLP_val_loss)
    time_file_name = time_result_path + '/MLP_M_time.csv'
    np.savetxt(time_file_name, run_time)


def MLP_hy_train(data_path, pos_path, config_path, model_result_path='./results/model',
                 loss_result_path='./results/loss'):
    config = util.load_config(config_path)
    dataset = util.load_data(data_path)
    pos_set = util.load_data(pos_path)
    criterion = nn.MSELoss()
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    train_set = dataset[:6000, : config['n_features'] * config['n_nodes']]
    val_set = dataset[6000: 8000, :config['n_features'] * config['n_nodes']]
    train_pos = pos_set[:6000, :config['n_features'] * config['n_nodes']]
    val_pos = pos_set[6000: 8000, :config['n_features'] * config['n_nodes']]
    train_set = Dataset(train_set, train_pos, config)
    val_set = Dataset(val_set, val_pos, config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False)

    net_p1 = models.MLP_M_part_1(config)
    net_p1.cuda()

    optimizer1 = optim.Adam(net_p1.parameters(), lr=config['learning_rate'])

    Hy_train_loss = []
    Hy_val_loss = []
    for epoch in tqdm(range(config['epochs'])):
        loss_temp = 0
        n_set = 0

        net_p1.train()
        for i, (features, targets, pos) in enumerate(train_loader):
            x = features.cuda()
            yr = targets.cuda()
            z = (-1 * (pos - 1)).cuda()
            y = net_p1(x, z)
            loss = criterion(y * z, yr * z)
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            loss_temp += loss.item()
            n_set += 1
        temp1 = loss_temp / n_set
        Hy_train_loss.append(temp1)

        net_p1.eval()
        with torch.no_grad():
            val_x = torch.FloatTensor(val_set.features).cuda()
            val_yr = torch.FloatTensor(val_set.targets).cuda()
            val_z = torch.FloatTensor(-1 * (val_set.pos - 1)).cuda()

            y = net_p1(val_x, val_z).detach()
            val_loss = criterion(y * val_z, val_yr * val_z)

        Hy_val_loss.append(val_loss.cpu().item())

        if (epoch + 1) % 100 == 0:
            print('Epoch[%d/%d], train_loss:%.8f, val_loss:%.8f'
                  % (epoch + 1, config['epochs'], temp1, Hy_val_loss[-1]))

    net_p1.cpu()

    net_p2 = models.MLP_M_part_2(config)
    model = models.MLP_hy(net_p1, net_p2)
    model.cuda()
    optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])

    for epoch in tqdm(range(config['epochs'])):
        loss_temp = 0
        n_set = 0

        model.train()
        for i, (features, targets, pos) in enumerate(train_loader):
            x = features.cuda()
            yr = targets.cuda()
            z = (-1 * (pos - 1)).cuda()
            y = model(x, z)
            loss = criterion(y * z, yr * z)
            loss.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            loss_temp += loss.item()
            n_set += 1
        temp1 = loss_temp / n_set
        Hy_train_loss.append(temp1)

        model.eval()
        with torch.no_grad():
            val_x = torch.FloatTensor(val_set.features).cuda()
            val_yr = torch.FloatTensor(val_set.targets).cuda()
            val_z = torch.FloatTensor(-1 * (val_set.pos - 1)).cuda()

            y = model(val_x, val_z).detach()
            val_loss = criterion(y * val_z, val_yr * val_z)

        Hy_val_loss.append(val_loss.cpu().item())
        if (epoch + 1) % 100 == 0:
            print('Epoch[%d/%d], train_loss:%.8f, val_loss:%.8f'
                  % (epoch + 1, config['epochs'], temp1, Hy_val_loss[-1]))

    model_file_name = model_result_path + '/MLP_hy_net.pth.tar'
    torch.save({'state_dict': model.state_dict()}, model_file_name)
    train_loss_file_name = loss_result_path + '/MLP_hy_train_loss.csv'
    np.savetxt(train_loss_file_name, Hy_train_loss)
    val_loss_file_name = loss_result_path + '/MLP_hy_val_loss.csv'
    np.savetxt(val_loss_file_name, Hy_val_loss)
