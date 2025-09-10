import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import time
import os
from PIL import Image
from data.datamgr import SetDataManager
from methods.protonet import ProtoNet
from methods.good_embed import GoodEmbed
from methods.meta_deepbdc import MetaDeepBDC
from methods.stl_deepbdc import STLDeepBDC
from utils import *
import argparse
import tqdm
from sklearn.metrics import f1_score
import cv2
from utils import *
from torchvision import transforms


def save_gradcam(image, mask, img_path, output_dir="gradcam_results"):
    """
    将 CAM 叠加到原图并保存
    """
    os.makedirs(output_dir, exist_ok=True)

    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)

    # 处理 mask
    # 1. 上采样到原始图像尺寸
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # 2. 转换为 uint8 类型
    mask_uint8 = np.uint8(255 * mask)

    # 3. 生成热力图
    heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # 4. 叠加热力图和原始图像
    cam = 0.4 * heatmap + 0.6 * np.float32(image)  # 调整叠加比例
    cam = cam / (cam.max() + 1e-8)  # 归一化

    # # 7. 调整分辨率到 800x800，使用双立方插值
    # # cam_resized = cv2.resize(cam, (800, 800), interpolation=cv2.INTER_CUBIC)  # 使用双立方插值
    # from PIL import Image
    # cam_pil = Image.fromarray(np.uint8(255 * cam))
    # cam_resized_pil = cam_pil.resize((800, 800), Image.ANTIALIAS)  # 使用抗锯齿方式调整大小
    # cam_resized = np.array(cam_resized_pil) / 255.0  # 转回 NumPy 并规范化

    # 保存结果
    img_name = os.path.basename(img_path).split('.')[0]
    cam_path = os.path.join(output_dir, f"{img_name}_gradcam.png")
    cv2.imwrite(cam_path, np.uint8(255 * cam))
    print(f"Saved Grad-CAM visualization to {cam_path}")


# 新增命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=84, type=int, choices=[84, 224])
parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet', 'tiered_imagenet', 'cub'])
parser.add_argument('--data_path', type=str)
parser.add_argument('--model', default='ResNet12', choices=['ResNet12', 'ResNet18'])
parser.add_argument('--method', default='stl_deepbdc',
                    choices=['meta_deepbdc', 'stl_deepbdc', 'protonet', 'good_embed'])

parser.add_argument('--test_n_way', default=5, type=int, help='number of classes used for testing (validation)')
parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=15, type=int,
                    help='number of unlabeled data in each class during meta validation')

parser.add_argument('--test_n_episode', default=2000, type=int, help='number of episodes in test')
parser.add_argument('--model_path', default='', help='meta-trained or pre-trained model .tar file path')
# parser.add_argument('--test_task_nums', default=5, type=int, help='test numbers')
parser.add_argument('--gpu', default='0', help='gpu id')

parser.add_argument('--penalty_C', default=0.1, type=float, help='logistic regression penalty parameter')
parser.add_argument('--reduce_dim', default=640, type=int,
                    help='the output dimensions of BDC dimensionality reduction layer')
parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')
parser.add_argument('--seed', default=1, type=int, help='random seed')

# 新增单图像测试参数
# parser.add_argument('--query_image_path', type=str, default='', help='Path to the single query image')
# parser.add_argument('--query_label', type=int, default=-1, help='True label index in support set (0~n_way-1)')
# 在参数解析中添加class_order参数
# parser.add_argument('--class_order', nargs='+', type=str,
#                     default=['burn_through', 'contamination', 'good_weld',
#                             'lack_of_fusion', 'lack_of_penetration', 'misalignment'],
#                     help='指定类别顺序（需与数据目录名称一致）')
parser.add_argument('--class_order', nargs='+', type=str,
                    default=['burn_through', 'contamination', 'good_weld',
                            'high_travel_speed', 'lack_of_fusion', 'lack_of_shielding_gas'],
                    help='指定类别顺序（需与数据目录名称一致）')
# parser.add_argument('--class_order', nargs='+', type=str,
#                     default=['burn_through', 'excessive_penetration', 'good_weld',
#                             'lack_of_penetration', 'weld_deviation', 'weld_misalignment'],
#                     help='指定类别顺序（需与数据目录名称一致）')

# 新增命令行参数：指定要生成热图的类索引
parser.add_argument('--class_idx', type=int, default=0, help='Index of the class to generate Grad-CAM for (0~n_way-1)')

params = parser.parse_args()
num_gpu = set_gpu(params)
# set_seed(params.seed)

# 加载查询图像的函数（支持多张查询图片）
def load_query_images(params, query_image_paths):
    """
        加载多个查询图像，每个类一张图片
        """
    data_transform = transforms.Compose([
        transforms.CenterCrop(700),
        transforms.ToTensor(),  # 只转换为 Tensor，不归一化
    ])

    from data.datamgr import TransformLoader
    transform_loader = TransformLoader(params.image_size)
    test_transform = transform_loader.get_composed_transform(aug=False)

    query_images1 = []
    query_images2 = []
    for img_path in query_image_paths:
        img = Image.open(img_path).convert('RGB')
        img1 = data_transform(img)
        img2 = test_transform(img)
        query_images1.append(img1)
        query_images2.append(img2.unsqueeze(0).cuda())  # [1, C, H, W]

    return query_images1, torch.cat(query_images2, dim=0)  # [n_way, C, H, W]

# 加载查询图像
query_image_paths = [
    './dataset/miniImageNet/0.png',  # 第 0 类的查询图片
    './dataset/miniImageNet/1.png',  # 第 1 类的查询图片
    './dataset/miniImageNet/2.png',  # 第 2 类的查询图片
    './dataset/miniImageNet/3.png',  # 第 3 类的查询图片
    './dataset/miniImageNet/4.png',  # 第 4 类的查询图片
    './dataset/miniImageNet/5.png'   # 第 5 类的查询图片
]

query_images1, query_images2 = load_query_images(params, query_image_paths)  # query_images2: [6, C, H, W]

# 数据加载器设置
json_file_read = False
if params.dataset == 'cub':
    novel_file = 'novel.json'
    json_file_read = True
else:
    novel_file = 'test'

# 指定图像测试模式：仅加载支持集
novel_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
novel_datamgr = SetDataManager(params.data_path, params.image_size, n_query=0,  # 关键修改：n_query=0
                               n_episode=params.test_n_episode, json_read=json_file_read, class_order=params.class_order,  # 传入类别顺序
                               **novel_few_shot_params)
novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)

# 模型初始化（保持不变）
if params.method == 'protonet':
    model = ProtoNet(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'good_embed':
    model = GoodEmbed(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'meta_deepbdc':
    model = MetaDeepBDC(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'stl_deepbdc':
    model = STLDeepBDC(params, model_dict[params.model], **novel_few_shot_params)

model = model.cuda()
model.eval()

# 加载模型
print(params.model_path)
model_file = os.path.join(params.model_path)
model = load_model(model, model_file)

# 定义钩子捕获目标层的特征和梯度
target_layer = None
if params.model == "ResNet12":
    # target_layer = model.feature.layer1[0].conv3  # ResNet12 的最后一个卷积层
    target_layer = model.feature.elav
elif params.model == "ResNet18":
    target_layer = model.feature.layer4[0].conv3  # ResNet18 的最后一个卷积层

# 存储特征和梯度
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])  # grad_output 是元组，取第一个元素

# 注册钩子
forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)

# 清空缓存
feature_maps.clear()
gradients.clear()

# 加载支持集数据
(x_support, _) = next(iter(novel_loader))  # x_support: [6, n_support, C, H, W]
x_support = x_support.cuda()

# 拼接支持图像和查询图片
x = torch.cat([x_support, query_images2.unsqueeze(1)], dim=1)  # [6, n_support + 1, C, H, W]

# 前向传播并计算梯度
model.n_query = 1
scores = model.set_forward(x, is_feature=False)
pred_class = scores.argmax(dim=1)[params.class_idx].item()  # 获取指定类的预测结果

# 反向传播获取梯度
model.zero_grad()
scores[params.class_idx, pred_class].backward(retain_graph=True)

# 计算 CAM
features = feature_maps[0].squeeze().cpu().detach().numpy()  # [6, n_support + 1, 640, 10, 10]
grads = gradients[0].squeeze().cpu().detach().numpy()  # [6, n_support + 1, 640, 10, 10]

# 计算查询图片的索引
query_index = params.class_idx * (params.n_shot + 1) + params.n_shot
# 只处理指定类的查询图片的特征图和梯度
query_features = features[query_index]  # 取出指定类的查询图片的特征图 [640, 10, 10]
query_grads = grads[query_index]  # 取出指定类的查询图片的梯度 [640, 10, 10]

# 计算权重
weights = np.mean(query_grads, axis=(1, 2))  # 梯度全局平均作为权重 [640]

# 计算 CAM
cam = np.zeros(query_features.shape[1:], dtype=np.float32)  # [10, 10]
for i, w in enumerate(weights):
    cam += w * query_features[i]  # 加权特征图
# cam = np.maximum(cam, 0)  # ReLU 激活
cam = np.abs(cam)  # 取绝对值
cam = cam / (cam.max() + 1e-8)  # 归一化

# 生成可视化
save_gradcam(query_images1[params.class_idx], cam, query_image_paths[params.class_idx])

print(f"Class {params.class_idx} | Predicted class: {pred_class}")



