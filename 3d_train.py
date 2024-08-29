import torch
import numpy as np



def fit_net(model, train_dataloader, test_dataloader, loss_fn, optim, exp_lr_scheduler):
    """
    返回每一个epoch的train_loss, train_acc, test_loss, test_acc, 返回四个标量值
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()  # 模型训练模式
    for x, y in train_dataloader:
        x, y = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
        x = x.to(device)
        y = y.to(device)
        y_train_pred = model(x)
        loss = loss_fn(y_train_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_train_pred = torch.argmax(y_train_pred, dim=1)  # 返回一个包含最大值索引位置的张量
            num_correct += (y_train_pred == y).sum().item()
            num_total += y.size(0)
            running_loss += loss.item()
    exp_lr_scheduler.step()
    train_acc = num_correct / num_total
    train_loss = running_loss / len(train_dataloader.dataset)

    num_correct_test = 0
    num_total_test = 0
    running_loss_test = 0
    model.eval()   # 模型验证模式或者测试模型
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            y_test_pred = model(x)
            loss = loss_fn(y_test_pred, y)
            y_test_pred = torch.argmax(y_test_pred, dim=1)
            num_correct_test += (y_test_pred == y).sum().item()
            num_total_test += y.size(0)
            running_loss_test += loss.item()

    test_acc = num_correct_test / num_total_test
    test_loss = running_loss_test / len(test_dataloader.dataset)

    return train_loss, train_acc, test_loss, test_acc

































