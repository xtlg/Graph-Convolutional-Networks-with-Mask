import copy
import torch
import time
import numpy as np
from tqdm import tqdm
from torch import optim, nn
from utils import util
from torchinfo import summary
from torch.utils.data import DataLoader


def model_train(model, model_name, train_set, val_set, adj_path, result_path):
    mat = util.get_adj(adj_path)
    adj = mat.cuda()
    summary(model)
    model.cuda()

    train_loss = []
    val_loss = []
    best_model = copy.deepcopy(model)
    best_loss = 1

    criterion = nn.MSELoss()
    dataloader = DataLoader(train_set, batch_size=256)
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    st = time.time()
    for epoch in tqdm(range(2000)):
        loss_temp = 0
        n_set = 0

        model.train()
        for (f, t, pos) in dataloader:
            x = f.cuda()
            yr = t.cuda()
            z = (-1 * (pos - 1)).cuda()

            y, _, _, _ = model(x, z, adj)
            loss = criterion(y * z, yr * z)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_temp += loss.item()
            n_set += 1
        temp1 = loss_temp / n_set
        train_loss.append(temp1)

        model.eval()
        with torch.no_grad():
            val_x = torch.FloatTensor(val_set.features).cuda()
            val_yr = torch.FloatTensor(val_set.targets).cuda()
            val_z = torch.FloatTensor(-1 * (val_set.pos - 1)).cuda()
            val_y, _, _, _ = model(val_x, val_z, adj)
            v_loss = criterion(val_y * val_z, val_yr * val_z)
            val_loss.append(v_loss.item())

        if (epoch + 1) % 100 == 0:
            print('Epoch[%d/%d], train_loss:%.8f, val_loss:%.8f'
                  % (epoch + 1, 2000, temp1, v_loss))

        if v_loss < best_loss:
            best_loss = v_loss
            best_model = copy.deepcopy(model)
    time_cost = np.array(time.time() - st).reshape(-1, 1)

    model_path = result_path + '/model/' + model_name + '.pth.tar'
    torch.save({'state_dict': best_model.state_dict()}, model_path)
    train_loss_name = result_path + '/loss/' + model_name + '_train_loss.csv'
    np.savetxt(train_loss_name, train_loss, delimiter=',')
    val_loss_name = result_path + '/loss/' + model_name + '_val_loss.csv'
    np.savetxt(val_loss_name, val_loss, delimiter=',')
    time_path = result_path + '/time/' + model_name + '_time.csv'
    np.savetxt(time_path, time_cost, delimiter=',')


def model_test(model, test_set, adj_path):
    mat = util.get_adj(adj_path)
    adj = mat.cuda()
    criterion = nn.MSELoss()
    model.cuda()
    model.eval()
    with torch.no_grad():
        val_x = torch.FloatTensor(test_set.features).cuda()
        val_yr = torch.FloatTensor(test_set.targets).cuda()
        val_z = torch.FloatTensor(-1 * (test_set.pos - 1)).cuda()
        val_y, _, _, _ = model(val_x, val_z, adj)
        v_loss = criterion(val_y * val_z, val_yr * val_z)

    return v_loss.item()
