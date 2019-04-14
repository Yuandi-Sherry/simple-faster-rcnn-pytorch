from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt

# 去正则化，img维度为[[B, G, R], H, W]
def inverse_normalize(img):
    if opt.caffe_pretrain:  # 如果采用caffe预训练模型，返回
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)) # 还原减去均值的预处理，并改为3*1*1的维度
        # items in the array reversed
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

# 预处理，输入像素值为0-1
def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    """
    使用channel = (channel - mean) / std进行归一化，将像素值转化为[-1,1]
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img)) # 转化为tensor并归一化
    return img.numpy()

# 采用caffe与训练模型对输入图像进行标准化，输入为0-1
def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1) # 转化维度
    img = (img - mean).astype(np.float32, copy=True) # 减去均值
    return img

# 输入图像0-255
def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape # 通道数、高度、宽度
    # 获得长边和短边的缩放倍数
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    # 采取较小倍数进行缩放
    img = img / 255. # 0-255 -> 0-1
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    # both the longer and shorter should be less than max_size and min_size
    # 选择一种正则化方法，在前面定义的两个函数
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)

# 变换类
class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape # 缩放后的图像大小
        scale = o_H / H # 缩放比例
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W)) # 按比例缩放包围盒

        # horizontally flip， 将图像随机水平翻转
        img, params = util.random_flip(img, x_random=True, return_param=True)
        # 将包围盒进行与图像相同的水平翻转
        bbox = util.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale

# 生成训练集
class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir) # 实例化类
        self.tsf = Transform(opt.min_size, opt.max_size) # 获得变换类

    def __getitem__(self, idx): # 运行Dataset自动运行
        # 调用VOCBboxDataset中的get_example()将img, bbox, label, difficult一个个取出
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        # 调用前面的Transform函数将图片, label进行最小值最大值放缩归一化
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)

# 生成测试集
class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
