import os
import random

import torch
from torch import nn
from time import time
import time
import  csv
import copy
from torch.optim import lr_scheduler
from torchvision import models
from torch.utils import data
import config,dataset1

def compute_params(model):
    """Compute number of parameters"""
    n_total_params = 0
    for name, m in model.named_parameters():
        n_elem = m.numel()
        n_total_params += n_elem
    return n_total_params

def set_parameter_requires_grad(model,feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name,num_class,feature_extract,use_pretrained=True):
    model_ft = None
    input_size=224
    if model_name == 'resneXt50':
        """Resnet101"""
        # model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_fc_in = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_fc_in, 1)
    elif model_name == 'densenet121':
    # elif model_name == 'densenet169':
        model_ft = models.densenet121(pretrained=use_pretrained)
        # model_ft = models.densenet169(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_classF_in = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_classF_in, 1, bias=True)
    elif model_name == 'googlenet':
        model_ft = models.googlenet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_fc_in = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_fc_in, 1)
    elif model_name == 'efficient':
        model_ft = EfficientNet.from_pretrained('efficientnet-b7')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_fc_in = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_fc_in, 1)
    elif model_name == 'sanet':
        model_ft = sa_resnet50()
        weights_dict = torch.load("./path/sa_resnet50.pth", map_location=device)['state_dict']
        print(model_ft.load_state_dict(weights_dict, strict=False))
        # 删除有关分类层
        num_fc_in = model_ft.fc.in_features
        del model_ft.fc
        model_ft.fc = nn.Linear(num_fc_in, 1)

        for name, para in model_ft.named_parameters():
            # 除head外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    elif model_name == 'TwoHeadRes':
        model_ft = TwoHeadResNext()
    else:
        print("Invalid model_name, exiting...")
        exit()
    return model_ft,input_size

def get_k_fold_data(k, image_dir):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    # if k1==0:#第一次需要打开文件
    file = open(image_dir, 'r', encoding='utf-8')
    # reader=csv.reader(file )
    imgs_ls = [] #所有数据的列表
    cols = [] #所有类标签的列表，按十
    for line in file.readlines():
        # if len(line):
        line = line[:-1]
        imgs_ls.append(line)
        col = line.split()[3]
        cols.append(col)
    file.close()
    #print(len(imgs_ls))
    # 把层的内容写进列表 col
    # 对col中的内容进行计数，获得每一类的名称对应个数的字典
    col_dict = {}
    for i in cols:
        col_dict[i] = col_dict.get(i, 0) + 1
    # p = eval(input('每层抽取的比例（小数）：'))
    p = 0.2
    # 获得每一类的名称对应抽取个数的字典
    col_p = {} #每类取100个
    for i in col_dict.keys():
        col_p[i] = int(round(col_dict[i] * p))  # round用来四舍五入，不加int结果会变成无数个p

    # 打乱样本
    # random.shuffle(imgs_ls)
    rest = imgs_ls.copy()
    # val_res = {} #存放取得的每一折数据对应的字典，折id:对应列表,验证结果字典
    # train_res = {} #训练结果字典
    for i in range(k):
        train_res = []
        val_res = []
        col_p1 = col_p.copy()
        random.shuffle(rest) #每次抽取之前都打乱数据
        rest1 = rest.copy()
        for j in rest1:
            if col_p1.get(j.split()[3], 0) > 0:
                col_p1[j.split()[3]] -= 1
                # del i[3]
                rest.remove(j) # 保证不会重复
                val_res.append(j)
        #取完一折的数据，放入字典中
        train_res = list(set(imgs_ls) - set(val_res))

        #print(avg)
        f1 = open('./test/singleKfold/trainm_k_{}.txt'.format(i+1), 'w')#'{}_{}_{}.png'.format(prefix, network, key)
        # f1 = open('./test/singleKfold/trainf_k_{}.txt'.format(i+1), 'w')#'{}_{}_{}.png'.format(prefix, network, key)
        f2 = open('./test/singleKfold/testm_k_{}.txt'.format(i+1), 'w')
        # f2 = open('./test/singleKfold/testf_k_{}.txt'.format(i+1), 'w')
        # writer1 = csv.writer(f1)
        # writer2 = csv.writer(f2)
        for ii,row in enumerate(train_res):
            if ii == 0:
                f1.write(row)
            else:
                f1.write('\n')
                f1.write(row)
        for ii,row in enumerate(val_res):
            if ii == 0:
                f2.write(row)
            else:
                f2.write('\n')
                f2.write(row)
        f1.close()
        f2.close()

# 存模型，状态，路径，模型名字
def save_model(state, save_path, name,k,mae,epoch):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save_name = os.path.join(save_path, name+'_{}foldTransferm.{}.{}.pth'.format(k,mae,epoch))
    save_name = os.path.join(save_path, name+'_{}foldTransferf.{}.{}.pth'.format(k,mae,epoch))
    # 保存一个序列化（serialized）的目标到磁盘，保存训练好的模型权重，优化器，epoch
    # state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_mae':best_mae,'epoch': epoch}
    torch.save(state, save_name)
    return save_name

def k_fold(k,model_name):
    #loss_acc_sum,train_acc_sum, test_acc_sum = 0,0,0
    Ktrain_min_l,Ktrain_min_mae,Ktrain_acc1_max_l,Ktrain_acc2_max_l = [],[],[],[]
    Ktest_min_mae,Ktest_acc1_max_l,Ktest_acc2_max_l = [],[],[]

    feature_extract = True
    for i in range(k):
        # 确定要用的数据
        # train_k = r"./test/singleKfold/trainm_k_{}.txt".format(i+1)
        train_k = r"./test/singleKfold/trainf_k_{}.txt".format(i+1)
        # test_k = r"./test/singleKfold/testm_k_{}.txt".format(i+1)
        test_k = r"./test/singleKfold/testf_k_{}.txt".format(i+1)

        model_ft, input_size = initialize_model(model_name, 1, feature_extract, use_pretrained=True)
        print('Loaded Model {}, #PARAMS={:3.2f}M'
                  .format(model_name, compute_params(model_ft) / 1e6))
        # print(model_ft)
        # 放在GPU上运行
        model_ft = model_ft.cuda()
        model_ft = torch.nn.DataParallel(model_ft,device_ids=[0,1])
        # model_ft.to(device)
        #修改train函数，使其返回每一批次的准确率，tarin_ls用列表表示
        train_dataset = dataset1.Dataset(train_k, phase='train', input_shape=opt.input_shape)
        trainloader = data.DataLoader(train_dataset,
                                      batch_size=opt.train_batch_size,  # train_batch_size = 8
                                      shuffle=True,
                                      num_workers=opt.num_workers)
        test_dataset = dataset1.Dataset(test_k, phase='test', input_shape=opt.input_shape)
        testloader = data.DataLoader(test_dataset,
                                     batch_size=opt.test_batch_size,  # test_batch_size = 4
                                     shuffle=True,
                                     num_workers=opt.num_workers)
        # 优化器设置
        # optimizer = torch.optim.Adam([{'params': model_ft.parameters()}],lr=1e-2, weight_decay=opt.weight_decay)
        optimizer = torch.optim.SGD([{'params': model_ft.parameters()}], lr=1e-2, momentum=opt.momentum)
        # 学习率调整策略
        # 每200个epoch调整一次，即整个训练过程中不降低
        scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)  # step_size
        # 先冻结前面所有卷积层，训练最后的分类器
        print(test_k)
        model_ft, filename, _, _, _, _, _, _, _ = train_model(model_name,i+1,model_ft, trainloader, testloader, optimizer, scheduler,
                                                              begin_ep=0,num_enpochs=50)
        # 再使用较小的学习率对整个网络进行微调，暂定两部分为：100+100epoch
        for param in model_ft.parameters():
            param.requires_grad = True
        # 优化器设置，减小学习率
        # optimizer = torch.optim.Adam([{'params': model_ft.parameters()}],lr=1e-4, weight_decay=opt.weight_decay)
        optimizer = torch.optim.SGD([{'params': model_ft.parameters()}], lr=1e-4, momentum=opt.momentum)
        # 学习率调整策略
        scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)  # step_size
        checkpoint = torch.load(filename)
        model_ft.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(test_k)
        model_ft, filename, loss_min,train_mae_min,train_acc1_max,train_acc2_max, \
        test_mae_min,test_acc1_max,test_acc2_max = train_model(model_name,i+1,model_ft, trainloader, testloader, optimizer, scheduler,
                                                               begin_ep=50,num_enpochs=50)
        checkpoint = torch.load(filename)
        best_toepoch = checkpoint['epoch']
        print(best_toepoch)
        # 求K次的平均值
        Ktrain_min_l.append(loss_min)
        Ktrain_min_mae.append(train_mae_min)
        Ktrain_acc1_max_l.append(train_acc1_max)
        Ktrain_acc2_max_l.append(train_acc2_max)
        Ktest_min_mae.append(test_mae_min)
        Ktest_acc1_max_l.append(test_acc1_max)
        Ktest_acc2_max_l.append(test_acc2_max)

    print('train_loss_min: %.4f,train_mae_min: %.4f, train_acc1_max: %.4f,train_acc2_max: %.4f,'
          'test_mae_min: %.4f, test_acc1_max: %.4f, test_acc2_max: %.4f'
          % (sum(Ktrain_min_l)/len(Ktrain_min_l), sum(Ktrain_min_mae)/len(Ktrain_min_mae), sum(Ktrain_acc1_max_l)/len(Ktrain_acc1_max_l), sum(Ktrain_acc2_max_l)/len(Ktrain_acc2_max_l),
             sum(Ktest_min_mae)/len(Ktest_min_mae), sum(Ktest_acc1_max_l)/len(Ktest_acc1_max_l), sum(Ktest_acc2_max_l)/len(Ktest_acc2_max_l)))
    # print("")
    # return sum(Ktrain_min_l)/len(Ktrain_min_l),sum(Ktrain_min_mae)/len(Ktrain_min_mae),sum(Ktrain_acc1_max_l)/len(Ktrain_acc1_max_l),sum(Ktrain_acc2_max_l)/len(Ktrain_acc2_max_l), \
    #        sum(Ktest_min_mae)/len(Ktest_min_mae),sum(Ktest_acc1_max_l)/len(Ktest_acc1_max_l),sum(Ktest_acc2_max_l)/len(Ktest_acc2_max_l)

def train_model(model_name,k,model,trainloader,testloader,optimizer,scheduler,begin_ep,num_enpochs=50,is_inception=False):
    since = time.time()
    best_acc = 0
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    """
    model.to(device)

    # 损失函数，取预测值和真实值的绝对误差的平均数
    criterion = torch.nn.L1Loss() #torch.nn.SmoothL1Loss
    train_loss,train_maeL,train_acc1L,train_acc2L = [],[],[],[]
    test_maeL,test_acc1L,test_acc2L = [],[],[] #求列表最大值最小值
    best_model_wts = copy.deepcopy(model.state_dict())
    bestTrain_mae = 100
    best_mae = 100 # 输出结果最好的mae
    filename = ""
    # result_file = r"./test/singleKfold/csv/" + model_name + "_{}koldm_".format(k) + "" + "Result.csv"
    result_file = r"./test/singleKfold/csv/" + model_name + "_{}koldf_".format(k) + "" + "Result.csv"
    with open(result_file, "a",newline='',encoding='utf-8')as f:
        b_csv = csv.writer(f)
        b_csv.writerow(["Epoch", "Loss", "MAEr", "Acc1r", "Acc2r", "MAEe", "Acc1e", "Acc2e"])  # writerow按行写入
        for epoch in range(begin_ep,begin_ep+num_enpochs):
            # 训练
            scheduler.step()
            model.train()
            loss,running_loss = 0.0,0.0
            train_correct1,train_correct2 = 0,0
            train_acc1,train_acc2 = 0,0
            train_num = 0
            train_mae = 0
            # running_corr = 0.0
            # 训练部分
            for ii, data in enumerate(trainloader):
                data_input, label, age_group1, age_group2 = data
                data_input = data_input.to(device)
                label = label.to(device).float().reshape([label.shape[0], 1])  # 8 TO 8，1
                output = model(data_input)
                # 计算损失
                loss = criterion(output, label)
                # 梯度清零
                optimizer.zero_grad()
                # 反向传播，计算梯度
                loss.backward()
                # 更新权重
                optimizer.step()
                # iters = epoch * len(trainloader) + ii + 1
                label = label.data.cpu().numpy()
                predicted = output.data.cpu().numpy()
                for i, age in enumerate(predicted):
                    # 计算十岁年龄段准确率
                    if (abs(age - label[i]) <= 10):
                        train_correct1 += 1
                    # 计算五岁年龄段准确率
                    if (abs(age - label[i]) <= 5):
                        train_correct2 += 1
                # 平均绝对误差是绝对误差的平均值。可以更好地反映预测值误差的实际情况。
                mae = float(sum(abs(predicted - label)) / len(predicted))
                train_mae += mae
                train_num += data_input.size(0)
                # item()是得到一个元素张量里面的元素值,具体就是：用于将一个零维张量转换成浮点数，比如计算loss，accuracy的值
                current_loss = loss.item()
                running_loss += current_loss #* data_input.size(0)
                # running_corr += torch.eq(np.floor_divide(predicted,10),age_group1).sum().item()
            train_mae /= (ii + 1)
            train_acc1 = train_correct1 / train_num
            train_acc2 = train_correct2 / train_num
            epoch_loss = running_loss / (ii + 1) #loss每个epoch输出一次
            #
            train_loss.append(epoch_loss)
            train_maeL.append(train_mae)
            train_acc1L.append(train_acc1)
            train_acc2L.append(train_acc2)
            # 总最佳准确率
            if train_mae < bestTrain_mae:
                bestTrain_mae = train_mae
            print('最佳训练MAE为: {:.4f}'.format(bestTrain_mae))
            # 每个epoch打印一次
            print('训练fold {} Epoch {} Loss {:.4f} MAE {:.4f} | Acc: {:.4f}|{:.4f}'.format(k, epoch + 1,
                                                                                          epoch_loss, train_mae,
                                                                                          train_acc1,
                                                                                          train_acc2))
            # print("训练准确率：{:.4f} | {:.4f}".format(train_acc1, train_acc2),train_correct1 ,"|",train_correct2 ,"|", train_num)
            # epoch_acc = running_corr / len(trainloader.dataset)
            # time_elapsed = time.time() - since
            # print("Time elaspsed {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            # print("{} Loss: {:.4f}".format('train',epoch_loss))
            # train_acc_history.append(epoch_acc)
            # train_loss.append(epoch_loss)
            print("Waiting Test!")
            # 测试部分
            with torch.no_grad():
                total_mae = 0
                test_correct1,test_correct2 = 0,0
                test_acc1,test_acc2 = 0,0
                total_num = 0
                for ii, data in enumerate(testloader):
                    model.eval()
                    images, labels, age_group1, age_group2 = data
                    images = images.to(device)
                    labels = labels.to(device).float().reshape([labels.shape[0], 1])

                    outputs = model(images)
                    labels = labels.data.cpu().numpy()
                    predicted = outputs.data.cpu().numpy()
                    for i, age in enumerate(predicted):
                        # 计算十岁年龄段准确率
                        if (abs(age - labels[i]) <= 10):
                            test_correct1 += 1
                        # 计算五岁年龄段准确率
                        if (abs(age - labels[i]) <= 5):
                            test_correct2 += 1
                    mae = float(sum(abs(predicted - labels)) / len(predicted))
                    total_mae += mae
                    total_num += images.size(0)
                total_mae /= (ii + 1)  # 平均???
                test_acc1 = test_correct1 / total_num
                test_acc2 = test_correct2 / total_num
                b_csv.writerow([epoch + 1,epoch_loss, round(train_mae,4), train_acc1, train_acc2, round(total_mae,4), test_acc1, test_acc2])
                test_maeL.append(total_mae)
                test_acc1L.append(test_acc1)
                test_acc2L.append(test_acc2)
                if total_mae < best_mae:
                    best_mae = total_mae
                    # 保存最后的模型
                    best_model_wts = copy.deepcopy(model.state_dict())
                    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_mae': best_mae,
                             'epoch': epoch + 1}
                    filename = save_model(state, r"./test/singleKfold/checkpoints", model_name,k,round(best_mae,4),epoch+1)
                # 每个epoch打印一次
                print('测试fold {} Epoch {} MAE {:.4f} | Acc: {:.4f}|{:.4f}'.format(k, epoch + 1,
                                                                                  total_mae,
                                                                                  test_acc1,
                                                                                  test_acc2))

    index_min = test_maeL.index(min(test_maeL))
    # f1 = open(r"./test/singleKfold/{}resultsM.txt".format(model_name), "a")
    f1 = open(r"./test/singleKfold/{}resultsF.txt".format(model_name), "a")
    if k == 1:
        f1.write("fold" + "  " + "train_loss" + "  " + "train_mae" + "  " + "train_acc1" + "  " + "train_acc2" + "  " + "test_mae" + "  " + "test_acc1"+ "  " + "test_acc2")
    f1.write('\n')
    f1.write('%d, train_loss: %.4f,train_mae: %.4f, train_acc1: %.4f,train_acc2: %.4f,'
          'test_mae: %.4f, test_acc1: %.4f, test_acc2: %.4f'
          % (k, train_loss[index_min], train_maeL[index_min], train_acc1L[index_min], train_acc2L[index_min],
          test_maeL[index_min],test_acc1L[index_min],test_acc2L[index_min]))
    f1.close()
    print('fold %d, train_loss_min: %.4f,train_mae_min: %.4f, train_acc1_max: %.4f,train_acc2_max: %.4f,'
          'test_mae_min: %.4f, test_acc1_max: %.4f, test_acc2_max: %.4f'
          % (k, train_loss[index_min], train_maeL[index_min], train_acc1L[index_min], train_acc2L[index_min],
          test_maeL[index_min],test_acc1L[index_min],test_acc2L[index_min]))
    time_elapsed = time.time() - since
    print("Time elaspsed {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    # 用最后的模型当做返回结果
    model.load_state_dict(best_model_wts)
    return model,filename,train_loss[index_min], train_maeL[index_min], train_acc1L[index_min], train_acc2L[index_min],\
           test_maeL[index_min],test_acc1L[index_min],test_acc2L[index_min]

if __name__ == '__main__':
    k = 5
    # image_dir = r"./test/singleKfold/allDataHm.txt"
    image_dir = r"./test/singleKfold/allDataHf.txt"
    # image_dir = r"./test/kfold/allDataF.txt"
    # 定义net，loss函数，优化器，训练周期等等
    opt = config.Config()
    # 展示结果，可视化
    gpu = '0'
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # 划分得到5折数据
    # get_k_fold_data(k, image_dir)  # 在main函数执行一次即可！
    # 执行k次交叉训练，k_fold会依次调用train和val函数
    print(image_dir)
    model_name = "resneXt50"  # opt.backbone   'resnet50' densenet121
    # print(model_name, ",{}折交叉验证m:".format(k))
    print(model_name, ",{}折交叉验证f:".format(k))
    k_fold(k,model_name)
