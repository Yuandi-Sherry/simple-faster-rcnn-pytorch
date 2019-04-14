from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from utils import array_tool as at
from utils.vis_tool import Visualizer

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        # 继承父模块的初始化
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma # 超参：在_faster_rcnn_loc_loss调用用来计算位置损失函数

        # target creator create gt_bbox gt_label etc as training targets. 
        # 用于从20000个候选anchor中产生256个anchor进行二分类和位置回归，也就是为rpn网络产生的预测位置和预测类别提供真正的ground_truth标准
        self.anchor_target_creator = AnchorTargetCreator()
        # AnchorTargetCreator和ProposalTargetCreator是为了生成训练的目标（或称ground truth），只在训练阶段用到，ProposalCreator是RPN为Fast R-CNN生成RoIs，在训练和测试阶段都会用到。所以测试阶段直接输进来300个RoIs，而训练阶段会有AnchorTargetCreator的再次干预。
        self.proposal_target_creator = ProposalTargetCreator()
        # (0., 0., 0., 0.)
        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        # (0.1, 0.1, 0.2, 0.2)
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer() # SGD
        # visdom wrapper
        self.vis = Visualizer(env=opt.env) # 可视化工具

        # indicators for training status
        # 混淆矩阵，验证预测值和真实值精确度，2为类别数
        self.rpn_cm = ConfusionMeter(2)
        #
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0] # batch个数
        print("trainer.py - batch个数", n)
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs) # vgg16 con5_3之前的部分提取图片特征

        # rpn_locs维度(hh*ww*9, 4), rpn_scores维度(hh*ww*9, 2), rois维度(2000,4), roi_indices用不到
        # anchor的维度为(hh*ww*9, 4), H和W是经过数据预处理之后的。计算(H/16)*(W/16)*9大概20000个anchor属于前景的概率
        # 取12000个经过NMS得到2000个近似目标框G^坐标。roi维度为(2000,4)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0] # (N, R, 4)
        label = labels[0] # (N, R)
        rpn_score = rpn_scores[0] #(hh * ww * 9, 4)
        rpn_loc = rpn_locs[0] # hh * ww * 9
        roi = rois # (2000, 4)

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # 调用proposal_target_creator函数生成sample roi(128,4),
        # gt_roi_loc(128, 4), gt_roi_label(128,1)

        # RoIHead网络利用sample_roi+featue为输入，
        # 输出是分类（21类）和回归（进一步微调bbox）的预测值，
        # 那么分类回归的ground truth就是ProposalTargetCreator输出的
        # gt_roi_label和gt_roi_loc
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        # 输入20000个anchor和bbox，调用anchor_target_creator函数
        # 得到2000个anchor与bbox的偏移量与label
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc, # rpn网络回归的偏移量20000个
            gt_rpn_loc, # anchor_target_creator函数得到的2000个anchor与bbox偏移量
            gt_rpn_label.data,
            self.rpn_sigma) # 1

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy( # 求交叉熵损失
            rpn_score, # rpn网络得到的20000个
            gt_rpn_label.cuda(), # anchor_target_creator得到的2000个label
            ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1] # 不计算背景类
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        # roi_cls_loc为VGG16RoIHead的输出(128*84)，n_sample=128
        n_sample = roi_cls_loc.shape[0]
        # roi_cls_los = (128, 21, 4)
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        # proposal_target_creator()生成的128个proposal与bbox求得偏移量dx, dy, dw, dh
        gt_roi_label = at.totensor(gt_roi_label).long()
        # 128个标签
        gt_roi_loc = at.totensor(gt_roi_loc)

        # 采用smooth_l1_loss
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        # 求交叉熵损失
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        # 四个loss加起来
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)


    # 参数优化
    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad() # 清零梯度
        losses = self.forward(imgs, bboxes, labels, scale) # 计算所有损失
        losses.total_loss.backward() # 反向传播计算梯度
        self.optimizer.step() # 参数更新
        self.update_meters(losses) # 将损失更新到可视化界面
        return losses # 返回损失

    # 保存模型
    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    # 加载模型
    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    # 更新可视化
    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    # 重置函数
    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    # 获取数据
    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


# rpn回归框偏移量与anchor偏移量，与bbox偏移量，label
def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss 除去背景类
    return loc_loss
