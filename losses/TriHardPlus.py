import torch
import torch.nn.functional as F
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x.float(), y.float().t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_similarity(x:torch.Tensor,y:torch.Tensor, eps:float=1e-12) -> torch.Tensor:
    """
    Computes cosine similarity between two tensors.
    Value == 1 means the same vector
    Value == 0 means perpendicular vectors
    """
    x_n, y_n = x.norm(dim=1)[:, None], y.norm(dim=1)[:, None]
    x_norm = x / torch.max(x_n, eps * torch.ones_like(x_n))
    y_norm = y / torch.max(y_n, eps * torch.ones_like(y_n))
    sim_mt = torch.mm(x_norm, y_norm.transpose(0, 1))
    return sim_mt


def cosine_dist(x:torch.Tensor,y:torch.Tensor,multi=None ,eps:float=1e-12) -> torch.Tensor:
    """
    Computes cosine distance between two tensors.
    The cosine distance is the inverse cosine similarity
    -> cosine_distance = abs(-cosine_distance) to make it
    similar in behaviour to euclidean distance
    """
    sim_mt = cosine_similarity(x,y, eps)
    if multi!=None:
        sim_mt=multi*sim_mt
    return torch.abs(1-sim_mt).clamp(min=eps)

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

class TriHardPLoss(object):  ###
    """如果强行要求 $d_{pn} > d_{ap}$，但 $p$ 和 $n$ 本身在全局分布中并不是一对真正的“难样本”，那么网络就会把宝贵的梯度浪费在优化一对原本就很安全的样本上，从而彻底破坏了难样本挖掘（Hard Mining）的初衷。为了解决这个“既要全空间约束，又要保留难度机制”的死结，我们不能使用静态权重的约束。我们需要引入一种极其优雅的机制：“难度感知动态路由（Difficulty-Aware Dynamic Routing）”"""

    def __init__(self, margin=None, dist_func='euclidean',weight_angular=0.1):
        self.margin = margin
        if margin is not None:
            # 必须加上 reduction='none'，为了后续进行基于难度的样本级加权
            self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')
        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction='none')

        if dist_func == 'cosine':
            self.dist_func = cosine_dist
        elif dist_func == 'euclidean':
            self.dist_func = euclidean_dist
        self.weight_angular=weight_angular
    def __call__(self, global_feat, labels, warmup_margin=False, print_data=False, current_epoch=None, max_epoch=None,
                 normalize_feature=False, mask=None, multi=0.0001, mutual_dist=False, factor=0.03, indx=1, last_gap=45):

        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)

        dist_mat = self.dist_func(global_feat, global_feat)

        if mutual_dist:
            self.dist_func_2 = cosine_dist
            dist_mat_cos = (0.55 * self.dist_func_2(global_feat, global_feat, multi=multi)).sqrt()
            dist_mat = dist_mat * (torch.log(1 + torch.exp(dist_mat_cos)))

        # # 依旧使用您写好的 hard_example_mining 提取最难的正负样本
        # dist_ap, dist_an, pdx, ndx = hard_example_mining(dist_mat, labels, current_epoch, max_epoch, factor=factor,
        #                                                  indx=indx, last_gap=last_gap, mask=mask, return_inds=True)
        dist_ap, dist_an, pdx, ndx = hard_example_mining(dist_mat, labels,return_inds=True)
        # 提取神秘的第三条边
        dist_pn = dist_mat[pdx, ndx]

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if mask is not None:
            dist_ap = dist_ap[mask]
            dist_an = dist_an[mask]
            dist_pn = dist_pn[mask]
            y = y[mask]

        # -------------------------------------------------------------------------
        # 【核心新增】：计算空间跷跷板的“难度感知动态权重” (Dynamic Routing Weights)
        # 我们使用 Softmin (即对负距离求 Softmax) 来比较 a-n 和 p-n 哪条边更短、更危险。
        # temperature 是温度系数，设为 5.0~10.0 可以让权重分配更加锐利。
        # -------------------------------------------------------------------------
        # temperature = 10.0
        # threat_logits = torch.stack([-dist_an, -dist_pn], dim=1) * temperature
        threat_logits = factor*torch.pow(torch.stack([-dist_an, -dist_pn], dim=1),indx)
        dynamic_weights = F.softmax(threat_logits.detach(), dim=1)

        # w_an: n 威胁 a 的程度 (常规权重)
        # w_pn: n 威胁 p 的程度 (跷跷板惩罚权重)
        w_an = dynamic_weights[:, 0]  # shape: [B]
        w_pn = dynamic_weights[:, 1]  # shape: [B]

        # 分别计算三个约束的 Raw Loss（此时由于 reduction='none'，它们保留了 [B] 的维度）
        if self.margin is not None:
            loss_an_raw = self.ranking_loss(dist_an, dist_ap, y)
            loss_pn_raw = self.ranking_loss(dist_pn, dist_ap, y)
        else:
            loss_an_raw = self.ranking_loss(dist_an - dist_ap, y)
            loss_pn_raw = self.ranking_loss(dist_pn - dist_ap, y)

        # 钝角约束 Raw Loss
        loss_angle_raw = torch.nn.functional.relu(dist_an ** 2 + dist_ap ** 2 - dist_pn ** 2)
        loss_angle_raw = torch.clamp(loss_angle_raw, max=2.0)
        # -------------------------------------------------------------------------
        # 【难度分配】：将危险度权重乘以对应的 Loss，再求平均
        # -------------------------------------------------------------------------
        # 如果 n 正常排斥，w_an 大，主导优化；
        loss_an = (w_an * loss_an_raw).mean()

        # 如果 n 试图缩小和 p 的距离，w_pn 瞬间飙升，强行激活对称约束和角度约束！
        loss_pn = (w_pn * loss_pn_raw).mean()
        # loss_angle = (w_pn * loss_angle_raw).mean()
        loss_angle = (loss_angle_raw).mean()
        # 最终组合：由于权重 w_an + w_pn = 1，相当于做了一个动态门控
        # loss = loss_an + loss_pn + loss_angle*0.2
        # loss = loss_an + loss_pn + loss_angle * 0.1
        if self.weight_angular > 1e-8:
            # from IPython import embed;embed();quit()
            loss =loss_an + loss_pn + loss_angle*self.weight_angular
            # loss = loss_an_raw.mean()+loss_angle*self.weight_angular
        else:
            loss = loss_an + loss_pn
            # loss = loss_an_raw.mean()
        if print_data:
            print(f'LOSS: {loss.item()}')
            prec = (dist_an > dist_ap).data.float().mean()
            print(f'precision: {prec}')
            print(f'AP mean distance: {dist_ap.data.mean()}')
            print(f'AN mean distance: {dist_an.data.mean()}')
            print(f'PN mean distance: {dist_pn.data.mean()}')
            # 打印一下权重分布，您可以直观看到网络是如何动态感知威胁的
            print(f'Mean Weight AN: {w_an.mean().item():.4f}, Mean Weight PN: {w_pn.mean().item():.4f}')

        return loss, dist_ap, dist_an, pdx, ndx