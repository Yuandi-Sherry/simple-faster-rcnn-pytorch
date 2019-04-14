from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        # 得到预测框位置、label和分数
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break
    # 完成预测水平的评估
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    # 将调用函数时候附加的参数用，config.py的opt._parse()解析，获取存储路径，放入dataset
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')

    # VOCBboxDataset作为数据读取库，读取图片，并调整和随机反转
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    # 数据装载到dataloader中，shuffle=True允许数据打乱，num_workers设置分批处理
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16() # 定义模型
    print('model construct completed')
    # 将FasterRCNNVGG16作为fasterrcnn的模型送入到FasterRCNNTrainer中
    # 并设置好GPU加速
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path: # 判断路径存在
        trainer.load(opt.load_path) # 读取与训练模型
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    # 开始训练，迭代次数在config.py预先定义，超参
    for epoch in range(opt.epoch):
        print ("---------------", epoch, " in ", opt.epoch, "-------------")
        trainer.reset_meters() # 可视化界面初始化数据
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            # 从训练数据中枚举dataloader，设置缩放范围，设置gpu加速
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # 调用trainer.py中的函数trainer.train_step(img, bbox, label, scale) 进行一次参数优化过程
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                # 判断数据读取次数是否能够整除plot_every，
                # 如果达到判断debug_file是否存在，用ipdb工具设置断点，
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss将训练数据读取并上传完成可视化
                trainer.vis.plot_many(trainer.get_meter_data())

                # 绘制Ground truth包围盒
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                # 将每次迭代读取的图片用dataset文件里面的inverse_normalize()
                # 函数进行预处理，将处理后的图片调用visdom_bbox
                trainer.vis.img('ground_truth_img', gt_img)

                # plot predict bboxes
                # 显示原始图片和预测结果（边框+类别）
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('predict_img', pred_img)

                # rpn confusion matrix(meter)
                # 调用trainer.vis.text将rpn_cm也就是RPN网络的混淆矩阵在可视化工具中显示
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        # 调用Trainer.vis.img将Roi_cm将roi的可视化矩阵以图片的形式显示
        trainer.vis.plot('test_map', eval_result['map'])
        # 设置学习率
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        # 将损失学习率以及map等信息及时显示更新
        trainer.vis.log(log_info)

        # 保存效果最好的map
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)

        # if判断句如果学习的epoch达到了9就将学习率*0.1变成原来的十分之一
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break # 结束训练过程


if __name__ == '__main__':
    import fire

    fire.Fire()
