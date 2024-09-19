import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from data import age_ary


def fit_net(model, train_dataloader, test_dataloader, test_dataset, loss_fn, optim, exp_lr_scheduler):
    """
    regression: 计算MAE/MSE/RMSE/MPE/R_2
    :param model: 初始化的模型
    :param train_dataloader: 训练集dataloader
    :param test_dataloader:  测试集dataloader
    :param loss_fn: 损失函数
    :param optim: 优化器
    :param lr_scheduler_epoch: 学习率的衰减，stepLR()
    :return: train_loss, train_acc, test_loss, test_acc。这是四个标量值
    """

    num_correct = 0
    num_total = 0
    running_loss = 0

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # model = torch.nn.DataParallel(model)
    model.train()  # 模型训练模式
    train_pred_list = []
    train_true_label = []
    for x, y in tqdm(train_dataloader):
        # x, y = torch.from_numpy(x), torch.from_numpy(y)
        x = x.to(device)
        y = y.to(device)
        y_train_pred = model(x)
        train_pred_list.append(y_train_pred.item())
        train_true_label.append(y.item())
        loss = loss_fn(y_train_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            # y_train_pred = torch.argmax(y_train_pred, dim=1)  # 返回一个包含最大值索引位置的张量
            # print("y_train_pred:", y_train_pred)
            # y_train_pred.item(),
    #         num_correct += (y_train_pred == y).sum().item()
    #         num_total += y.size(0)
            running_loss += loss.item()
    exp_lr_scheduler.step()
    # train_acc = num_correct / num_total
    train_loss = running_loss / len(train_dataloader.dataset)

    train_pred_age = np.array(train_pred_list)
    train_true_age = np.array(train_true_label)
    train_MAE = mean_absolute_error(train_true_age, train_pred_age)
    train_MSE = mean_squared_error(train_true_age, train_pred_age)
    train_RMSE = np.sqrt(train_MSE)
    train_MAPE = mean_absolute_percentage_error(train_true_age, train_pred_age)
    train_R2 = r2_score(train_true_age, train_pred_age)

    num_correct_test = 0
    num_total_test = 0
    running_loss_test = 0

    model.eval()   # 模型验证模式或者测试模型
    pred_list = []
    true_label = []
    with torch.no_grad():
        for x, y in tqdm(test_dataloader):
            x = x.to(device)
            y = y.to(device)
            y_test_pred = model(x)
            loss = loss_fn(y_test_pred, y)
            # y_test_pred = torch.argmax(y_test_pred, dim=1)
            # num_correct_test += (y_test_pred == y).sum().item()
            # num_total_test += y.size(0)
            running_loss_test += loss.item()
            pred_list.append(y_test_pred.item())
            true_label.append((y.item()))
    pred_age = np.array(pred_list)
    true_label = np.array(true_label)
    # age_ary = test_dataset.dataset.get_all_labels().numpy()  # tensor -> numpy
    # test_acc = num_correct_test / num_total_test
    test_loss = running_loss_test / len(test_dataloader.dataset)
    MAE = mean_absolute_error(true_label, pred_age)
    MSE = mean_squared_error(true_label, pred_age)
    RMSE = np.sqrt(MSE)
    MAPE = mean_absolute_percentage_error(true_label, pred_age)
    R2 = r2_score(true_label, pred_age)

    return train_loss, test_loss, MAE, MSE, RMSE, MAPE, R2, pred_age, true_label, \
        train_pred_age, train_true_age, train_MAE, train_MSE, train_RMSE, train_MAPE, train_R2


































