"""
编码器模块：统一的编码器接口和实现

主要特点：
- TorchGeo模型本质上是对timm的封装，统一在一个文件中实现
- 支持灵活的通道输入配置（不限制4+1通道）
- 标准化的接口设计，便于扩展
"""

from .encoders import DualStreamEncoder, create_encoder

__all__ = ['DualStreamEncoder', 'create_encoder']
