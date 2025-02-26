
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, dataset

import numpy as np


# from fitnet import fit_net
from data import get_pixel

from sklearn.model_selection import KFold

import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from data import age_ary


class Modified3DUNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=4):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        # self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
        #                              bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)

        self.linear = nn.Linear(32*28*28*28, 1)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        # out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)  # [64, 14 , 28, 28]

        # Level 1 localization pathway
        # out = torch.cat([out, context_4], dim=1)
        # out = self.conv_norm_lrelu_l1(out)
        # out = self.conv3d_l1(out)
        # out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)
        #
        # # Level 2 localization pathway
        # out = torch.cat([out, context_3], dim=1)
        # out = self.conv_norm_lrelu_l2(out)
        # ds2 = out
        # out = self.conv3d_l2(out)
        # out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)
        #
        # # Level 3 localization pathway
        # out = torch.cat([out, context_2], dim=1)
        # out = self.conv_norm_lrelu_l3(out)
        # ds3 = out
        # out = self.conv3d_l3(out)
        # out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        #
        # # Level 4 localization pathway
        # out = torch.cat([out, context_1], dim=1)
        # out = self.conv_norm_lrelu_l4(out)
        # out_pred = self.conv3d_l4(out)
        #
        # ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        # ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        # ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        # ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        # ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)
        #
        # out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        # # seg_layer = out
        # out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
        # # out = out.view(-1, self.n_classes)
        # # out = self.softmax(out)
        # out = out.T
        out = out  # [1, 32,28,28,28]
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




def get_imgs():
    with open("imgpath.txt", "r") as f:
        allimgs = f.readlines()
        f.close()
    image3d_path = []
    for item in allimgs:
        item = item.split("\n")[0]
        # print(item)
        image3d_path.append(item)
    return image3d_path


image3d_path = get_imgs()
# print(len(image3d_path))  # 1085
# label_ary = np.array(sex_ary, dtype=np.float32)
# label_ary = torch.from_numpy(age_ary).long()
label_ary = torch.from_numpy(age_ary)
# print("$$$$$$$$$$$$$$$$$$$$$$$")
# print(np.unique(label_ary))  # 01234567


class SkullDs(Dataset):
    def __init__(self, image_list, label_ary):
        self.image_list = image_list
        self.label_list = label_ary
        # self.transform = transformer

    def __getitem__(self, item):
        image = self.image_list[item]
        label = self.label_list[item]

        image3d = get_pixel(image)

        # image3d = self.transform(image3d)
        image3d = np.expand_dims(image3d, 0)
        # label = self.transform(label)
        print(f"img:{image}")
        with open("name_sort.txt", "a+") as f:
            f.write(image)


        return image3d, label

    def __len__(self):
        return len(self.label_list)


    def get_all_labels(self):
        return self.label_list


######################
#  待重新定义

# train_imgs = image3d_path[:1000]
# train_label = label_ary[:1000]

# test_imgs = image3d_path[1000:]
# test_label = label_ary[1000:]
##############################
# 待重新定义

# train_ds = SkullDs(train_imgs, train_label)
# test_ds = SkullDs(test_imgs, test_label)

# train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
# test_dl = DataLoader(test_ds, batch_size=1)
##################################

# img, label = next(iter(train_dl))
# print(img.shape, label.shape)
# print(type(img), type(label))

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = Modified3DUNet(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.0005)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    EPOCH = 60
    best_mae = 8
    k = 10

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    imgs = image3d_path
    labels = label_ary

    data = SkullDs(imgs, labels)  # 所有的图像和标签都打包成了dataset

    fold = 1
    for train_index, val_index in kf.split(data):
        print(f"fold:{fold}")
        train_ds = dataset.Subset(data, train_index)
        val_ds = dataset.Subset(data, val_index)
        # print("###:::", val_ds.dataset.get_all_labels().numpy())  # 索引
        train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)

        fold_acc_train = 0
        fold_acc_val = 0
        for epoch in tqdm(range(EPOCH)):
            train_loss, test_loss, MAE, MSE, RMSE, MAPE, R2, pred_age, true_label, \
                train_pred_age, train_true_age, train_MAE, train_MSE, train_RMSE, train_MAPE, train_R2 = fit_net(model,
                                                                                                                 train_dl,
                                                                                                                 val_dl,
                                                                                                                 val_ds,
                                                                                                                 loss_fn,
                                                                                                                 optimizer,
                                                                                                                 exp_lr_scheduler)
            # fold_acc_train += train_acc
            # fold_acc_val += test_acc
            # every epoch:
            output = f"fold:{fold},epoch:{epoch + 1}/{EPOCH}, " \
                     f"train_loss:{train_loss:.4f}, test_loss:{test_loss:.4f}, " \
                     f"test_MAE:{MAE}, test_MSE:{MSE}, test_RMSE:{RMSE}, test_MAPE:{MAPE}, test_R2_score:{R2}, " \
                     f"test_pred_age:{pred_age}, \ntest_true_age:{true_label}, " \
                     f"\ntrain_pred_age:{train_pred_age}, \ntrain_true_age:{train_true_age}, " \
                     f"\ntrain_MAE:{train_MAE:.4f}, train_MSE:{train_MSE:.4f}, train_RMSE:{train_RMSE:.4f}, " \
                     f"train_MAPE:{train_MAPE:.4f}, train_R2:{train_R2:.4f}"

            with open("contrast_3dUnet_age.txt", "a+") as f:
                f.write(output + "\n")

            if MAE < best_mae:
                best_mae = MAE
                # checkpoint = {
                #     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                #     "fold": fold, "lr": exp_lr_scheduler.state_dict()
                # }
                torch.save(model.state_dict(), "contrast_3dUnet_age.pth")

        fold += 1






