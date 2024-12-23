import os
import clip  # 导入CLIP库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from PIL import Image  # 导入PIL库，用于图像处理
import torchvision.transforms as transforms  # 导入TorchVision库中的变换工具
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize  # 导入常用的图像预处理方法
import numpy as np  # 导入NumPy库
try:
    from torchvision.transforms import InterpolationMode  # 尝试导入图像插值模式
    BICUBIC = InterpolationMode.BICUBIC  # 设置双三次插值模式
except ImportError:
    BICUBIC = Image.BICUBIC  # 如果旧版本的TorchVision没有InterpolateMode，则使用PIL的BICUBIC插值模式

# 定义ViT模型封装类
class vit_img_wrap(nn.Module):
    def __init__(self, clip_model='ViT-L/14', device='cpu', center=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
        super().__init__()

        # 加载CLIP模型和预处理方法
        self.model, self.preprocess = clip.load(clip_model, device)
        self.name = '-'.join(clip_model.split('/'))  # 解析模型名称
        self.device = device  # 设置设备（CPU或GPU）
        self.dim = self.model.text_projection.shape[1]  # 获取文本投影的维度，用于后续的特征转换
        self.inv_normalize = _transform(self.model.visual.input_resolution, center, std)  # 获取反向归一化变换

    def forward(self, image):
        # 前向传播，输入图像并返回特征
        image = self.inv_normalize(image)  # 对输入图像进行反向归一化
        with torch.no_grad():
            image_features = self.model.encode_image(image)  # 使用CLIP模型对图像进行编码，提取图像特征

        return image_features.float()  # 返回图像特征并转换为float类型

# 定义适配器模块，用于转换特征维度
class Adapter(nn.Module):
    """
    An adapter module for transforming feature dimensions.
    
    Attributes:
    - fc: A sequential neural network for feature transformation.
    """
    def __init__(self, dim):
        """
        Initialize the adapter with the given feature dimension.
        
        Parameters:
        - dim: The dimension of the input and output features.
        """
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),  # 全连接层，将输入维度转换为相同维度
            nn.Softplus(),  # 激活函数，用于非线性变换
            nn.Linear(dim, dim),  # 再次使用全连接层
            nn.Softplus(),  # 激活函数
        )

    def forward(self, x):
        """
        Forward pass through the adapter.
        
        Parameters:
        - x: The input features.
        
        Returns:
        - x: The transformed features.
        """
        x = self.fc(x)  # 前向传播，通过适配器的全连接层
        return x


# 定义CLIP图像适配器类，将CLIP图像编码器和适配器组合
class clip_img_adapter(nn.Module):
    """
    A class combining the CLIP image encoder and the adapter.
    
    Attributes:
    - clip_encoder: The CLIP image encoder.
    - adapter: The adapter module for feature transformation.
    - device: The device to run the model on.
    """
    def __init__(self, device='cuda'):
        """
        Initialize the CLIP image adapter with the given device.
        
        Parameters:
        - device: The device to run the model on.
        """
        super().__init__()

        # 初始化CLIP图像编码器
        self.clip_encoder = vit_img_wrap(clip_model='ViT-L/14', device=device)
        # 初始化适配器，转换特征维度
        self.adapter = Adapter(dim=768)  # 假设ViT-L/14的输出维度是768
        self.device = device
        # 将CLIP编码器和适配器移动到指定设备
        self.clip_encoder.to(device)
        self.clip_encoder.eval()  # 设置为评估模式
        self.adapter.to(device)
        self.adapter.eval()  # 设置为评估模式

    def forward(self, image):
        """
        Forward pass through the CLIP image encoder and the adapter.
        
        Parameters:
        - image: The input image.
        
        Returns:
        - feature: The transformed image features.
        """
        with torch.no_grad():
            # 通过CLIP编码器提取图像特征
            feature = self.clip_encoder(image)
            # 通过适配器变换特征维度
            feature = self.adapter(feature)

        return feature  # 返回适配后的特征

# 定义图像预处理方法
def _transform(n_px, center=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """
    Create a composition of transformations for preprocessing images.
    
    Parameters:
    - n_px: The size to resize the images to.
    - center: The mean values for normalization.
    - std: The standard deviation values for normalization.
    
    Returns:
    - A Compose object containing the transformations.
    """
    return Compose([
        Normalize(mean=[-center[0] / std[0], -center[1] / std[1], -center[2] / std[2]],  # 将中心值归一化
                  std=[1 / std[0], 1 / std[1], 1 / std[2]]),  # 将标准差归一化
        Resize(n_px, interpolation=BICUBIC),  # 调整图像大小，使用BICUBIC插值
        CenterCrop(n_px),  # 中心裁剪图像
        Normalize((0.48145466, 0.4578275, 0.40821073),  # 进行标准化处理，减去训练集的均值
                  (0.26862954, 0.26130258, 0.27577711)),  # 使用训练集的标准差
    ])
