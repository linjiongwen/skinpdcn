import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score  # 新增F1计算依赖
from .template import MetaTemplate
from .bdc_module import BDC
from ELA import * ##

class MetaDeepBDC(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(MetaDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        
        reduce_dim = params.reduce_dim
        self.feat_dim = int(reduce_dim * (reduce_dim+1) / 2)
        self.dcov = BDC(is_vec=True, input_dim=self.feature.feat_dim, dimension_reduction=reduce_dim)

    def feature_forward(self, x):
        out = self.dcov(x)
        return out

    def set_forward(self, x, is_feature=False):

        # 特征提取[原]
        z_support, z_query = self.parse_feature(x, is_feature)
        # 原型计算（每个类别的支持集特征均值）[原]
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        # 查询集特征展平[原]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        # 计算相似度
        scores = self.metric(z_query, z_proto)

        # 添加形状断言[改]
        assert scores.shape == (self.n_way * self.n_query, self.n_way), \
            f"输出形状错误，应为 [{(self.n_way * self.n_query)}, {self.n_way}]，实际为 {scores.shape}"
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        scores = self.set_forward(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)
        

        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores

    def metric(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        if self.n_support > 1:
            dist = torch.pow(x - y, 2).sum(2)
            score = -dist
        else:
            score = (x * y).sum(2)
        return score
        

