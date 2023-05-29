from utils import test_tool
import torch

torch.manual_seed(1)
torch.cuda.manual_seed(1)

GCNM_test = test_tool.GCN_test(model_file_path='./results/model/GCN_mask_True_net.pth.tar',
                               data_path='./data/39Bus.csv',
                               test_file_path='./data/test_data',
                               adj_path='./data/adj.xls',
                               config_path='./config/config_mask.yaml',
                               test_result_path='./results/test/GCNM')

GCN_test = test_tool.GCN_test(model_file_path='./results/model/GCN_mask_False_net.pth.tar',
                              data_path='./data/39Bus.csv',
                              test_file_path='./data/test_data',
                              adj_path='./data/adj.xls',
                              config_path='./config/config_no_mask.yaml',
                              test_result_path='./results/test/GCN')

MLP_test = test_tool.MLP_test(model_file_path='./results/model/MLP_net.pth.tar',
                              data_path='./data/39Bus.csv',
                              test_file_path='./data/test_data',
                              config_path='./config/config_mask.yaml',
                              test_result_path='./results/test/MLP')

generator_test = test_tool.generator_test(model_file_path='./results/model/W_generator_net.pth.tar',
                                          data_path='./data/39Bus.csv',
                                          test_file_path='./data/test_data',
                                          config_path='./config/config_WGAN.yaml',
                                          test_result_path='./results/test/GAN')

MLP_M_test = test_tool.MLP_M_test(model_file_path='./results/model/MLP_M_net.pth.tar',
                                  data_path='./data/39Bus.csv',
                                  test_file_path='./data/test_data',
                                  config_path='./config/config_mask.yaml',
                                  test_result_path='./results/test/MLP_M')


GCNM_test_under_noise = test_tool.GCN_test_under_noise(model_file_path='./results/model/GCN_mask_True_net.pth.tar',
                                                       data_path='./data/39Bus.csv',
                                                       test_file_path='./data/test_data',
                                                       adj_path='./data/adj.xls',
                                                       config_path='./config/config_mask.yaml',
                                                       test_result_path='./results/test/GCNM')
