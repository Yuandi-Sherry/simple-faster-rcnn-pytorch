import numpy as np
import cupy as cp

from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox
from model.utils.nms import non_maximum_suppression

# 在RoIHead实现，为2000个rois赋予ground truth
# 输入2000个rois，一张图中所有的bbox ground truth(R,4)，对应bbox包含的Label(R,1)
# 输出128个sample roi(128 * 4), 128个gt_roi_loc(128,4), 128个gt_roi_label(128,1)
class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs.

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    # 接受ProposalCreator产生的2000个ROIS，但是这些ROIS并不都用于训练
    # 经过本ProposalTargetCreator的筛选产生128个用于自身的训练
    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """Assigns ground truth to sampled proposals.

        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.

        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.

        Args:
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bouding boxes.
            loc_normalize_std (tupler of four floats): Standard deviation of
                the coordinates of bounding boxes.

        Returns:
            (array, array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        n_bbox, _ = bbox.shape
        # 将2000个roi和m个bbox相连成为新的roi(2000+m, 4)
        roi = np.concatenate((roi, bbox), axis=0)
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        # 计算每个roi与每个bbox的iou
        iou = bbox_iou(roi, bbox)
        # 按行找最大值，返回最大值对应的下标以及其IOU。返回每个roi与bbox最大，以及最大的iou值
        gt_assignment = iou.argmax(axis=1)
        # 每个roi与对应bbox最大的iou
        max_iou = iou.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        # 从1开始的类别序号，给每个类得到真正的label
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        # 从1开始类别序号，给每个类得到真正label 0-19 -> 1-20
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        # 根据iou的最大值将正负样本找出
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        # 需要保留的roi个数，64及以下，随机丢弃
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        # 需要保留roi个数（满足大于0小于neg_iou_thresh_hi条件的roi与64之间较小
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

# 这里输出的128*4的sample_roi可以扔到RoIHead网络里面进行分类回归。同样，RoIHead网络利用这sample_roi+feature为输入，输出分别是分类（21类）和回归的预测值，分类回归的ground truth为ProposalTargetCreator输出的gt_roi_label和gt_roi_loc
        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        # 求128个样本的ground truth
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))
        # 归一化
        return sample_roi, gt_roi_loc, gt_roi_label


#在RPN网络实现，生成训练使用的anchor（与对应框iou值最大或者最小的各128个框的坐标和256个label(0或1)）
class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.
    将ground truth的包围盒赋值给训练区域的anchors

    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.
    函数model.utils.bbox_tools.bbox2loc计算匹配ground truth需要的偏移量和缩放量
    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.

    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.

        Types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array):

            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.

        """

        img_H, img_W = img_size

        n_anchor = len(anchor) # 对应20000个足有anchor
        inside_index = _get_inside_index(anchor, img_H, img_W) # 将超范围的anchor去掉，保留图片内部的序号
        anchor = anchor[inside_index] # 保留位于图片内部的anchor
        argmax_ious, label = self._create_label(inside_index, anchor, bbox) # 筛选出符合条件的正例128个负例128个附上label

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious]) # 计算每个anchor与对应的bbox求得iou最大的bbox计算偏移量

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1) # 将位于图片内部的框label对应到所生成的20000个框中（label原本为所有图片中的框）
        loc = _unmap(loc, n_anchor, inside_index, fill=0) # 将回归的框对应到所生成的20000个框中

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)
        # 得到每个anchor与哪个bbox的iou最大以及iou值 todo 体会行和列取最大值的区别
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)
        # 把每个anchor与对应的框求得iou值与负样本阈值比较，小于负样本阈值，则label为0，
        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0
        # 把与每个bbox求得iou值最大的anchor的label设为1
        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1

        # 把每个anchor与对应的框求得的iou值与正样本阈值比较，若大于正样本阈值，则label设为1
        label[max_ious >= self.pos_iou_thresh] = 1

        # 按比例计算正样本数量
        n_pos = int(self.pos_ratio * self.n_sample)
        # 得到所有正样本的索引
        pos_index = np.where(label == 1)[0]
        # 如果选取出来的正样本太多，随机抛弃，将抛弃的label设为-1
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # 设定负样本数量
        n_neg = self.n_sample - np.sum(label == 1)
        # 找到负样本索引
        neg_index = np.where(label == 0)[0]
        # 如果负样本数量太多
        if len(neg_index) > n_neg:
            # 随机选择不要的负样本
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # 计算anchor与bbox的IOU，N个anchor，K个bbox
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1) # 1代表行，0代表列
        max_ious = ious[np.arange(len(inside_index)), argmax_ious] # 求出每个anchor与哪个bbox的iou最大，以及最大值，max_ious:[1, N]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])] # 求出每个bbox与哪个anchor的iou最大，以及最大值，gt_max_ious:[1,k]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0] # 返回最大iou索引（有k个）

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


# 在RPN网络实现，生成regions
class ProposalCreator:
    # 对于每张图片，利用它的feature map，计算（H/16）x(W/16)x9(大概20000)个anchor
    # 属于前景的概率，然后从中选取概率较大的12000张，利用位置回归参数，
    # 修正这12000个anchor的位置，
    # 利用非极大值抑制，选出2000个ROIS以及对应的位置参数。

    """Proposal regions are generated by calling this object.

    The :meth:`__call__` of this object outputs object detection proposals by
    applying estimated bounding box offsets
    to a set of anchors.

    This class takes parameters to control number of bounding boxes to
    pass to NMS and keep after NMS.
    If the paramters are negative, it uses all the bounding boxes supplied
    or keep all the bounding boxes returned by NMS.

    This class is used for Region Proposal Networks introduced in
    Faster R-CNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        nms_thresh (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in test mode.
        force_cpu_nms (bool): If this is :obj:`True`,
            always use NMS in CPU mode. If :obj:`False`,
            the NMS mode is selected based on the type of inputs.
        min_size (int): A paramter to determine the threshold on
            discarding bounding boxes based on their sizes.

    """

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size


    # 这里的loc和score是经过region_proposal_network中经过1x1卷积分类和回归得到的
    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        """input should  be ndarray
        Propose RoIs.

        Inputs :obj:`loc, score, anchor` refer to the same anchor when indexed
        by the same index.

        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.

        Type of the output is same as the inputs.

        Args:
            loc (array): Predicted offsets and scaling to anchors.
                Its shape is :math:`(R, 4)`.
            score (array): Predicted foreground probability for anchors.
                Its shape is :math:`(R,)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(R, 4)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The scaling factor used to scale an image after
                reading it from a file.

        Returns:
            array:
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.

        """
        # NOTE: when test, remember
        # faster_rcnn.eval()
        # to set self.traing = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms # 12000
            n_post_nms = self.n_train_post_nms # 2000
        else:
            n_pre_nms = self.n_test_pre_nms # 6000
            n_post_nms = self.n_test_post_nms # 300

        # Convert anchors into proposal via bbox transformations.
        roi = loc2bbox(anchor, loc)

        # Clip predicted boxes to image.
        # 裁剪将rois的ymin,ymax限定在[0,H]
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        # 裁剪将rois的xmin,xmax限定在[0,W]
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0] # rois的宽
        ws = roi[:, 3] - roi[:, 1] # rois的长
        keep = np.where((hs >= min_size) & (ws >= min_size))[0] # 确保rois的长宽大于最小阈值
        roi = roi[keep, :]
        score = score[keep] # 对剩下的ROIs进行打分（根据region_proposal_network中rois的预测前景概率）

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order = score.ravel().argsort()[::-1] # score拉伸并且排序
        if n_pre_nms > 0:
            order = order[:n_pre_nms] #训练取前1200，test取前6000
        roi = roi[order, :]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        keep = non_maximum_suppression(
            cp.ascontiguousarray(cp.asarray(roi)),
            thresh=self.nms_thresh) # NMS原理以及输入参数的作用，将重复的抑制掉，train得到2000，test得到300
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi
