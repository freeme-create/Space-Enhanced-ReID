import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    """
    终极版 CenterLoss 主控端 (单质心完全向下兼容版)：
    - use_ema=False: 完全等价于原始的无归一化 SGD 单质心。
    - use_ema=True: 开启球面 L2 约束和 EMA 动量滑动更新。
    """
    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True, use_ema=False, ema_alpha=0.01):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha

        # 默认每个类只有一个质心: 形状由 [num_classes, num_proxies, feat_dim] 降维为 [num_classes, feat_dim]
        tensor = torch.randn(self.num_classes, self.feat_dim)
        if self.use_gpu:
            tensor = tensor.cuda()

        self.centers = nn.Parameter(tensor)
        nn.init.kaiming_normal_(self.centers, mode='fan_out')

        # ==========================================
        # 兼容性设计 1：初始化与梯度分流
        # ==========================================
        if self.use_ema:
            # EMA 模式：质心必须在超球面上，且关掉 SGD 梯度
            with torch.no_grad():
                self.centers.data = F.normalize(self.centers.data, p=2, dim=-1)
            self.centers.requires_grad = False
        else:
            # 原始模式：质心在自由空间，交还给 PyTorch 默认 SGD 优化
            self.centers.requires_grad = True

    def forward(self, x, labels, centrom_flag=False, reduce_vram_by_detach=True):
        batch_size = x.size(0)

        # ==========================================
        # 兼容性设计 2：特征计算空间的分流
        # ==========================================
        if self.use_ema:
            # EMA 模式：严格 L2 归一化 (流形约束)
            x_calc = F.normalize(x, p=2, dim=-1)
        else:
            # 原始模式：完全使用原始输入特征，等价于原版代码
            x_calc = x

        centers_for_dist = self.centers
        if reduce_vram_by_detach:
            centers_for_dist = centers_for_dist.detach()

        # 计算欧氏距离平方, distmat 形状: [B, num_classes]
        distmat = torch.pow(torch.cdist(x_calc, centers_for_dist), 2)

        # ==========================================
        # 1. 提取当前样本与对应 正质心 的距离 (d_ap)
        # ==========================================
        dist_Center_ap = distmat[torch.arange(batch_size), labels]  # [B]

        loss = dist_Center_ap.clamp(min=1e-12, max=1e+12).mean()

        # ==========================================
        # EMA 动量更新 (取代 SGD)
        # ==========================================
        if self.use_ema and self.training:
            with torch.no_grad():
                for i in range(batch_size):
                    c = labels[i]
                    # 动量滑动更新公式： C = (1 - alpha) * C + alpha * F
                    self.centers.data[c] = (1 - self.ema_alpha) * self.centers.data[c] + self.ema_alpha * x_calc[i].detach()
                # 更新完毕后，必须再次将质心拉回超球面
                self.centers.data = F.normalize(self.centers.data, p=2, dim=-1)

        if centrom_flag:
            # ==========================================
            # 2. 寻找最具威胁的 负质心 (d_an)
            # ==========================================
            distmat_neg = distmat.clone()
            distmat_neg[torch.arange(batch_size), labels] = float('inf')

            dist_notCenter_an_min, neg_c_idx = torch.min(distmat_neg, dim=1)  # [B]

            # ==========================================
            # 3. 计算 正质心 与 最硬负质心 之间的距离 (d_pn)
            # ==========================================
            centers_for_pn = self.centers if not reduce_vram_by_detach else self.centers.detach()

            pos_proxy_feats = centers_for_pn[labels]       # [B, D]
            neg_proxy_feats = centers_for_pn[neg_c_idx]    # [B, D]

            dist_ap2an = torch.pow(torch.norm(pos_proxy_feats - neg_proxy_feats, p=2, dim=1), 2)  # [B]

            return loss, self.centers, (dist_Center_ap, dist_notCenter_an_min, dist_ap2an)

        return loss, self.centers

class CentroidMLoss(nn.Module):
    def __init__(self, margin=None, num_class=751, use_gpu=True, num_proxies=3):
        super(CentroidMLoss, self).__init__()
        self.num_classes = num_class
        self.num_proxies = num_proxies
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_ap, dist_an, dist_pn,
                 current_epoch=None, max_epoch=None, factor=3, indx=1, last_gap=45):

        randN = 0.25
        if current_epoch != None and max_epoch != None:
            if current_epoch >= max_epoch - last_gap:
                randN = 0.5
            if torch.rand(1).data < randN:
                dist_an = torch.exp(factor * (-dist_pn + dist_an) ** indx ) * dist_pn / (
                        torch.exp(factor * (-dist_an + dist_an) ** indx ) + torch.exp(
                    factor * (-dist_pn + dist_an) ** indx )) + \
                          torch.exp(factor * (-dist_an + dist_an) ** indx ) * dist_an / (
                                  torch.exp(factor * (-dist_an + dist_an) ** indx ) + torch.exp(
                              factor * (-dist_pn + dist_an) ** indx ))

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin is not None:
            loss_metric = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss_metric = self.ranking_loss(dist_an - dist_ap, y)


        return loss_metric
