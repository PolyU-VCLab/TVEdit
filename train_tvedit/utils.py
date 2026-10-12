import json
import os
from PIL import Image, ImageDraw
import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from PIL import Image
import numpy as np
from skimage.morphology import dilation, disk
import random
import torch
from numba import njit, prange
from scipy.ndimage import binary_dilation
import io
def center_crop(image, target_width, target_height):
    """
    对图片进行中心裁剪
    
    参数:
        image: PIL.Image 对象，原始图片
        target_width: 裁剪后的宽度
        target_height: 裁剪后的高度
    
    返回:
        PIL.Image 对象，裁剪后的图片
    """
    # 获取原图尺寸
    original_width, original_height = image.size
    
    # 检查目标尺寸是否合法（不能超过原图尺寸）
    if target_width > original_width or target_height > original_height:
        raise ValueError("目标尺寸不能超过原图尺寸")
    
    # 计算裁剪区域的左上角坐标
    x = (original_width - target_width) // 2
    y = (original_height - target_height) // 2
    
    # 计算裁剪区域的右下角坐标（左闭右开区间）
    x2 = x + target_width
    y2 = y + target_height
    
    # 执行裁剪
    cropped_image = image.crop((x, y, x2, y2))
    return cropped_image
def remap_coordinates(coordinates, original_size, inplace=False):

    # 计算缩放因子

    coordinates[:,0] /= original_size[0]
    coordinates[:,0] *= 512

    coordinates[:,1] /= original_size[1]
    coordinates[:,1] *=512
    return coordinates

def create_mask(src_points,tgt_points):
    src_points = torch.tensor(src_points)
    tgt_points = torch.tensor(tgt_points)
    disp = src_points-tgt_points
    disp = disp.pow(2).sum(1)
    index= torch.nonzero(disp>0.05,as_tuple=False).squeeze()
    points = torch.cat((src_points[index],tgt_points[index]),dim=0)
    mask = generate_mask_with_convex_hull(points)
    return Image.fromarray(mask)


def generate_mask_with_convex_hull(points, image_size=(512, 512)):
    """
    根据点集生成一个不规则形状的mask，确保不超过原图尺寸。

    参数:
        points (list of tuples): 坐标点列表 [(x1, y1), ...]
        image_size (tuple): 图像大小 (width, height)

    返回:
        mask (np.ndarray): 二值mask (0/1)，类型为 uint8
    """
    # 确保points是numpy数组格式
    points = np.array(points)
    
    # 创建空白mask
    mask = np.zeros(image_size, dtype=np.uint8)
    try:
        # 计算凸包
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        # 确保坐标在图像范围内
        hull_points[:, 0] = np.clip(hull_points[:, 0], 0, image_size[1] - 1)
        hull_points[:, 1] = np.clip(hull_points[:, 1], 0, image_size[0] - 1)
        
        # 绘制多边形轮廓
        rr, cc = polygon(hull_points[:, 1], hull_points[:, 0], shape=image_size)
        mask[rr, cc] = 1
    except Exception as e:
        print(f"无法创建凸包: {e}")
    mask = fast_random_dilation(mask)
    return mask

def sample_points_from_two_frames(point1,point2):
    point1=torch.tensor(point1, dtype=torch.float32)
    point2=torch.tensor(point2, dtype=torch.float32)
    disp = (point1-point2).pow(2).sum(1)

    _,index=torch.topk(disp, 10, sorted=True)
    points = torch.cat((point1[index],point2[index]),dim=1)
    return points

from scipy import ndimage
from skimage import morphology

def fast_random_dilation(mask, max_kernel_size=30, prob=0.5):
    """
    对二值掩码进行随机膨胀，模拟区域扩展
    
    参数:
        mask (np.ndarray): 输入的二值掩码 (0/1)
        max_kernel_size (int): 最大膨胀核尺寸
        prob (float): 进行膨胀的概率
    
    返回:
        np.ndarray: 扩展后的二值掩码
    """
    if np.random.random() > prob:
        return mask.copy()
    
    # 随机选择膨胀核大小
    kernel_size = np.random.randint(1, max_kernel_size + 1)
    
    # 随机选择膨胀核形状（正方形或圆形）
    if np.random.random() > 0.5:
        # 正方形核
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    else:
        # 圆形核
        kernel = morphology.disk(kernel_size // 2)
    
    # 执行膨胀操作
    dilated_mask = ndimage.binary_dilation(mask, structure=kernel).astype(mask.dtype)
    
    return dilated_mask