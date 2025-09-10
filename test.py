import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import time
import os
from data.datamgr import SetDataManager

from methods.protonet import ProtoNet
from methods.good_embed import GoodEmbed
from methods.meta_Skinpbdcn import MetaDeepBDC
from methods.stl_deepbdc import STLDeepBDC

from utils import *
import argparse
import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc###
import matplotlib
matplotlib.use('Agg')  # 强制使用无 GUI 后端
import matplotlib.pyplot as plt###
from sklearn.metrics import f1_score #####
from sklearn.metrics import recall_score ####

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=84, type=int, choices=[84, 224])
parser.add_argument('--dataset', default='Skin_Cancer', choices=['Skin_Cancer', 'tiered_imagenet', 'cub'])
parser.add_argument('--data_path', type=str)
parser.add_argument('--model', default='ResNet12', choices=['ResNet12', 'ResNet18'])
parser.add_argument('--method', default='stl_deepbdc', choices=['meta_Skinpbdcn', 'stl_deepbdc', 'protonet', 'good_embed'])

parser.add_argument('--test_n_way', default=2, type=int, help='number of classes used for testing (validation)')
parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')

parser.add_argument('--test_n_episode', default=2000, type=int, help='number of episodes in test')
parser.add_argument('--model_path', default='', help='meta-trained or pre-trained model .tar file path')
parser.add_argument('--test_task_nums', default=10, type=int, help='test numbers')
parser.add_argument('--gpu', default='0', help='gpu id')

parser.add_argument('--penalty_C', default=0.1, type=float, help='logistic regression penalty parameter')
parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimensions of BDC dimensionality reduction layer')
parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')


params = parser.parse_args()
num_gpu = set_gpu(params)

json_file_read = False
if params.dataset == 'cub':
    novel_file = 'novel.json'
    json_file_read = True
else:
    novel_file = 'test'


novel_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
novel_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.test_n_episode, json_read=json_file_read,  **novel_few_shot_params)
novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)
# 在 novel_loader 创建后添加-定位问题
try:
    # 尝试从 DataLoader 的数据集中提取 transforms
    dataset = novel_loader.dataset
    if hasattr(dataset, 'transform'):
        print("测试集预处理配置:", dataset.transform)
    else:
        print("警告：无法直接获取 transforms，请检查数据集类定义")
except AttributeError:
    print("DataLoader 未提供 dataset 属性")

if params.method == 'protonet':
    model = ProtoNet(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'good_embed':
    model = GoodEmbed(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'meta_Skinpbdcn':
    model = MetaDeepBDC(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'stl_deepbdc':
    model = STLDeepBDC(params, model_dict[params.model], **novel_few_shot_params)

# model save path
model = model.cuda()
model.eval()

print(params.model_path)
model_file = os.path.join(params.model_path)
model = load_model(model, model_file) # 模型加载

print(params)
iter_num = params.test_n_episode
acc_all_task = []
f1_all_task = []  # 用于存储每个任务的 F1-score
# 在测试循环前初始化收集容器
y_true_all = []
y_pred_all = []
y_prob_all = []

for _ in range(params.test_task_nums):
    acc_all = []
    f1_all = []  # 用于存储当前任务的 F1-score
    test_start_time = time.time()
    tqdm_gen = tqdm.tqdm(novel_loader)
    for _, (x, _) in enumerate(tqdm_gen):
        with torch.no_grad():
            model.n_query = params.n_query
            scores = model.set_forward(x, False)  # 接收注意力权重
        
        # 获取概率和预测标签
        probabilities = torch.softmax(scores, dim=1) # 计算softmax概率
        prob = probabilities[:, 1].cpu().numpy() # 正类（malignant）的概率/prob模型对测试数据集中每个样本属于正类的概率。
        pred = torch.argmax(scores, dim=1).cpu().numpy()  # 预测标签
        # 假设数据加载器中第一个类别为 malignant（1），第二个为 benign（0）
        # 在测试循环中生成标签
        y = np.repeat([0, 1], params.n_query)  # 确保与数据顺序匹配
        
        #计算准确率
        acc = np.mean(pred == y) * 100
        f1 = f1_score(y, pred, average='binary', pos_label=1) * 100

        # 收集数据
        y_true_all.extend(y)
        y_pred_all.extend(pred)
        y_prob_all.extend(prob)
        acc_all.append(acc)
        f1_all.append(f1)
    
        # 更新进度条
        tqdm_gen.set_description(f'avg.acc:{(np.mean(acc_all)):.2f} curr.acc:{acc:.2f} | avg.f1:{(np.mean(f1_all)):.2f} curr.f1:{f1:.2f}')

    # 转换为 NumPy 数组
    acc_all = np.asarray(acc_all)
    f1_all = np.asarray(f1_all)
    total_samples = len(y_true_all)  # 实际总样本量
    # 计算 Accuracy 和 F1-score 的均值和标准差
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    f1_mean = np.mean(f1_all)
    f1_std = np.std(f1_all)
    # 打印结果，包括 Accuracy 和 F1-score
    current_task_samples = len(y) * len(tqdm_gen)  # 当前任务的样本量 = batch数 * 每批样本数
    print('%d Test Acc = %4.2f%% +- %4.2f%% | F1-score = %4.2f +- %4.2f (Time uses %.2f minutes)'
      % (current_task_samples, acc_mean, 1.96 * acc_std / np.sqrt(current_task_samples), 
         f1_mean, 1.96 * f1_std / np.sqrt(current_task_samples),(time.time() - test_start_time) / 60)) 

    # 记录每个任务的 Accuracy 和 F1-score
    acc_all_task.append(acc_all)
    f1_all_task.append(f1_all)

# 最终指标计算
y_true = np.array(y_true_all)
y_pred = np.array(y_pred_all)
y_prob = np.array(y_prob_all)

# 在所有任务结束后绘制图表
assert len(y_true_all) > 0, "y_true_all 为空！"
assert len(y_pred_all) > 0, "y_pred_all 为空！"
assert len(y_prob_all) > 0, "y_prob_all 为空！"
print(f"总样本量: {len(y_true_all)}")

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['benign', 'malignant'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('/root/Skinpbdcn/result-gram/confusion_matrix.png') # 设置保存路径
plt.close()

# ROC曲线
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('/root/Skinpbdcn/result-gram/roc_curve.png') # 指定保存路径
plt.close()

# 打印全局样本
print("Global prob min/max:", np.min(y_prob_all), np.max(y_prob_all))

# 最终指标打印
final_acc = np.mean(y_true == y_pred) * 100
final_f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0) * 100
print(f'\n{"="*50}\nFinal Metrics\n{"="*50}')
print(f'Total Samples: {len(y_true)}')
print(f'Accuracy: {final_acc:.2f}%')
print(f'F1-Score: {final_f1:.2f}%')
if 'roc_auc' in locals():
    print(f'AUC: {roc_auc:.2f}')

# 计算召回率
recall = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
print(f'Recall: {recall:.2f}%')