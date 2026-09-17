"""KuroSiwo洪水后图像分割数据集加载器。

这是一个全球SAR洪水制图数据集，仅使用洪水后的图像进行分割任务。

数据集结构：
- 使用pickle文件存储样本信息（train_pickle, test_pickle）
- 仅使用洪水后图像（MS1_IVV, MS1_IVH）进行分割
- 支持多种通道配置：vv, vh, vh/vv等
- 支持DEM和坡度数据作为辅助特征

参考：https://arxiv.org/abs/2311.12056
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2 as cv
import einops
import numpy as np
import pyjson5 as json
import torch
import torchvision
from compress_pickle import load
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# 导入现有的项目依赖
try:
    from torchgeo.datasets import NonGeoDataset
    BaseDataset = NonGeoDataset
except Exception:
    from torch.utils.data import Dataset as BaseDataset


def get_grids(pickle_path: str) -> Dict:
    """从pickle文件加载样本信息。"""
    if not os.path.isfile(pickle_path):
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")
    with open(pickle_path, "rb") as f:
        return load(f)


class KuroSiwoDataset(BaseDataset):
    """KuroSiwo洪水后图像分割数据集类

    仅使用洪水后的SAR图像进行洪水分段任务。

    Args:
        root: 数据根目录路径
        split: 数据集划分 ("train", "val", "test")
        channels: 通道配置，如 ["vv", "vh"] 或 ["vv", "vh", "vh/vv"]
        scale_input: 标准化方式 ("normalize", "min-max", "custom", "log")
        data_mean: 数据均值（用于标准化）
        data_std: 数据标准差（用于标准化）
        clamp_input: 输入裁剪阈值
        dem: 是否使用DEM数据作为辅助特征
        slope: 是否使用坡度数据（需要同时启用dem）
        dem_mean: DEM均值
        dem_std: DEM标准差
        slope_mean: 坡度均值
        slope_std: 坡度标准差
        uint8: 是否转换为uint8格式
        train_acts: 训练集激活ID列表
        val_acts: 验证集激活ID列表
        test_acts: 测试集激活ID列表
        transforms: 额外的变换
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        channels: List[str] = None,
        scale_input: str = "normalize",
        data_mean: List[float] = None,
        data_std: List[float] = None,
        data_min: Optional[List[float]] = None,
        data_max: Optional[List[float]] = None,
        clamp_input: float = 0.15,
        dem: bool = False,
        slope: bool = False,
        dem_mean: float = 93.4313,
        dem_std: float = 1410.8382,
        slope_mean: float = 2.1277,
        slope_std: float = 67.5048,
        dem_scale_mode: str = "zscore",  # "zscore" or "none"
        uint8: bool = False,
        train_acts: List[int] = None,
        val_acts: List[int] = None,
        test_acts: List[int] = None,
        transforms: Optional[Any] = None,
        **kwargs
    ) -> None:
        super().__init__()

        # 基础配置
        self.root = root
        self.split = split
        self.channels = channels or ["vv", "vh"]
        self.scale_input = scale_input
        # 保持与 DataModule 一致：允许在 log 模式传入 None，仅做 log1p 不做 z-score
        self.data_mean = data_mean
        self.data_std = data_std
        self.clamp_input = clamp_input
        # DEM/坡度配置（当前仅支持 DEM；slope 预留）
        self.dem = bool(dem)
        self.slope = bool(slope)
        self.dem_mean = dem_mean
        self.dem_std = dem_std
        self.slope_mean = slope_mean
        self.slope_std = slope_std
        # DEM 归一化方式：zscore(使用 dem_mean/std) 或 none(仅填补NaN，不缩放)
        self.dem_scale_mode = str(dem_scale_mode or "zscore").lower()
        self.uint8 = uint8
        self.transforms = transforms
        # 可选的自定义 min/max（当 scale_input == "custom" 时使用）
        self.data_min = torch.tensor(data_min, dtype=torch.float32) if data_min is not None else None
        self.data_max = torch.tensor(data_max, dtype=torch.float32) if data_max is not None else None

        # 数据集划分（使用官方激活ID）
        if train_acts is None:
            train_acts = [130, 470, 555, 118, 174, 324, 421, 554, 427, 518, 502,
                         498, 497, 496, 492, 147, 267, 273, 275, 417, 567,
                         1111011, 1111004, 1111009, 1111010, 1111006, 1111005]
        if val_acts is None:
            val_acts = [514, 559, 279, 520, 437, 1111003, 1111008]
        if test_acts is None:
            test_acts = [321, 561, 445, 562, 411, 1111002, 277, 1111007, 205, 1111013]

        self.train_acts = train_acts
        self.val_acts = val_acts
        self.test_acts = test_acts

        # 根据split设置有效激活列表
        if self.split == "train":
            self.valid_acts = self.train_acts
        elif self.split == "val":
            self.valid_acts = self.val_acts
        else:  # test
            self.valid_acts = self.test_acts

        # 统计信息
        self.clz_stats = {1: 0, 2: 0, 3: 0}
        self.act_stats = {}

        # 设置数据路径
        self._setup_data_paths()

        # 加载样本（仅通过官方pickle，不再扫描）
        self._load_samples()

        print(f"KuroSiwo {split} dataset (flood-only): {len(self.samples)} samples")
        print(f"Climate zones distribution: {self.clz_stats}")
        print(f"Activation distribution: {self.act_stats}")

    def _setup_data_paths(self) -> None:
        """设置数据路径"""
        # 强制使用 <root>/<events>/... 布局
        self.data_root = self.root

    def _get_pickle_path(self) -> str:
        """返回当前 split 对应的官方 V2 pickle 路径（位于 <root>/pickle）。"""
        pickle_dir = os.path.join(self.root, "pickle")
        if self.split == "train":
            return os.path.join(pickle_dir, "KuroV2_grid_dict.gz")
        else:
            return os.path.join(pickle_dir, "KuroV2_grid_dict_test_0_100.gz")

    def _resolve_sample_dir(self, relative_path: str) -> str:
        """将pickle中的相对路径解析为实际样本目录。
        优先尝试 <root>/<relative_path>，若不存在，再尝试 <root>/data/<relative_path>。
        """
        cand1 = os.path.join(self.data_root, relative_path)
        if os.path.isdir(cand1):
            return cand1
        cand2 = os.path.join(self.data_root, "data", relative_path)
        if os.path.isdir(cand2):
            return cand2
        return cand1  # 返回原始候选，后续访问会报错，便于定位问题

    def _load_samples(self) -> None:
        """从官方 V2 pickle 加载样本，并筛除无掩码样本。"""
        pickle_path = self._get_pickle_path()
        if not os.path.isfile(pickle_path):
            raise FileNotFoundError(
                f"未找到官方pickle: {pickle_path}. 请将 KuroV2_grid_dict*.gz 放在 <root>/pickle 下"
            )

        grids = get_grids(pickle_path)

        self.samples = []
        filtered_count = 0
        for key in grids:
            record = grids[key]
            rel_path = record["path"]
            info = record["info"]
            clz = record.get("clz", 1)
            activation = info.get("actid")

            if activation is None:
                continue

            if self.valid_acts and (activation not in self.valid_acts):
                continue

            # 掩码判定：以磁盘上是否存在 MK0_MLU*.tif 为准（元数据仅作参考）
            datasets = info.get("datasets", {})
            has_mask_meta = any(k.startswith("MK0_MLU") for k in datasets.keys())
            # 检查文件系统（兼容 <root>/<path> 与 <root>/data/<path>）
            dir1 = os.path.join(self.data_root, rel_path)
            dir2 = os.path.join(self.data_root, "data", rel_path)
            has_mask_file = False
            for base in (dir1, dir2):
                if os.path.isdir(base):
                    try:
                        files = os.listdir(base)
                        if any(fn.startswith("MK0_MLU") and fn.endswith(".tif") for fn in files):
                            has_mask_file = True
                    except Exception:
                        pass

            if not has_mask_file:
                filtered_count += 1
                continue  # 筛除无掩码样本

            resolved_path = rel_path  # 存相对路径（脚本依赖），真实读取时再解析

            self.clz_stats[clz] = self.clz_stats.get(clz, 0) + 1
            self.act_stats[activation] = self.act_stats.get(activation, 0) + 1

            self.samples.append({
                "id": key,
                "path": resolved_path,
                "info": info,
                "clz": clz,
                "activation": activation,
            })
        
        # 静默过滤统计，避免训练日志噪声

    def _get_augmentations(self):
        """占位：增强由 DataModule 使用 kornia 统一处理。"""
        return None

    def _concat_channels(self, image1: np.ndarray, image2: np.ndarray) -> torch.Tensor:
        """连接通道"""
        image1_exp = np.expand_dims(image1, 0)  # vv
        image2_exp = np.expand_dims(image2, 0)  # vh

        if set(self.channels) == set(["vv", "vh", "vh/vv"]):
            eps = 1e-7
            image = np.vstack(
                (image1_exp, image2_exp, image2_exp / (image1_exp + eps))
            )  # vv, vh, vh/vv
        elif set(self.channels) == set(["vv", "vh"]):
            image = np.vstack((image1_exp, image2_exp))  # vv, vh
        elif self.channels == ["vh"]:
            image = image2_exp  # vh
        elif self.channels == ["vv"]:
            image = image1_exp  # vv
        else:
            # 默认使用vv, vh
            image = np.vstack((image1_exp, image2_exp))

        image = torch.from_numpy(image).float()

        if self.clamp_input is not None:
            image = torch.clamp(image, min=0.0, max=self.clamp_input)
            image = torch.nan_to_num(image, self.clamp_input)
        else:
            image = torch.nan_to_num(image, 200)

        return image

    def _scale_image(self, img: torch.Tensor, valid_mask: torch.Tensor,
                     img_name: str, activation: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """图像标准化
        返回: (x1, x2, scaled)
        - normalize: x1=means(C,), x2=stds(C,), scaled=(img-means)/stds
        - min-max/custom: x1=mins(C,), x2=maxs(C,), scaled=(img-mins)/(maxs-mins)
        - log: 先对 img 执行 log1p，再按 data_mean/data_std 标准化（若提供）
        - db: 先对 img 执行 10*log10(max(img, eps))，第三通道为 dB 比值 (vh_db - vv_db)，再按 data_mean/data_std 标准化（若提供）
        """
        # 确保 valid_mask 为 HxW 布尔
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        C, H, W = img.shape

        def _align_stats(stats: Optional[List[float]], name: str) -> torch.Tensor:
            """将 data_mean / data_std 长度对齐到当前通道数 C，避免 C 与 len(stats) 不一致导致 Normalize 报错。

            - 若长度不足：重复最后一个值补齐
            - 若长度过长：截断
            """
            if stats is None:
                return torch.zeros(C, device=img.device, dtype=img.dtype) if name == "mean" else torch.ones(C, device=img.device, dtype=img.dtype)
            vals = list(stats)
            if len(vals) < C:
                vals = vals + [vals[-1]] * (C - len(vals))
            elif len(vals) > C:
                vals = vals[:C]
            return torch.tensor(vals, device=img.device, dtype=img.dtype)

        if self.scale_input == "normalize":
            means = _align_stats(self.data_mean, "mean")
            stds = _align_stats(self.data_std, "std")
            scaled = torchvision.transforms.Normalize(means, stds)(img)
            return means, stds, scaled

        elif self.scale_input == "min-max":
            # 按通道在有效像素上计算 min/max；若设置 clamp_input，则将 maxs 钳制到该上限
            mins = torch.empty(C, device=img.device, dtype=img.dtype)
            maxs = torch.empty(C, device=img.device, dtype=img.dtype)
            scaled = img.clone()
            for c in range(C):
                ch = img[c]
                if valid_mask.any():
                    ch_vals = ch[valid_mask]
                else:
                    ch_vals = ch.view(-1)
                min_c = ch_vals.min()
                max_c = ch_vals.max()
                if self.clamp_input is not None:
                    max_c = torch.minimum(max_c, torch.tensor(self.clamp_input, device=img.device, dtype=img.dtype))
                mins[c] = min_c
                maxs[c] = max_c
                if max_c > min_c:
                    scaled[c] = (ch - min_c) / (max_c - min_c)
                else:
                    scaled[c] = ch
            return mins, maxs, scaled

        elif self.scale_input == "custom":
            # 使用外部提供的每通道 min/max（data_min/data_max）
            if self.data_min is None or self.data_max is None:
                # 无自定义统计时，退化为 normalize（与官方“自定义配置缺失”时的容错一致性）
                means = _align_stats(self.data_mean, "mean")
                stds = _align_stats(self.data_std, "std")
                scaled = torchvision.transforms.Normalize(means, stds)(img)
                return means, stds, scaled
            mins = self.data_min.to(img.device, dtype=img.dtype)
            maxs = self.data_max.to(img.device, dtype=img.dtype)
            # 广播到 CHW
            denom = torch.clamp(maxs - mins, min=1e-12)
            scaled = (img - mins.view(-1, 1, 1)) / denom.view(-1, 1, 1)
            return mins, maxs, scaled

        elif self.scale_input == "log":
            # 对幅度取 log1p（就地计算以减少内存拷贝）
            img = img.clamp_min_(0)
            img.log1p_()
            # 若启用了三通道 [vv, vh, vh/vv]，则将第3通道改为“真正的对数比值”：log1p(vh) - log1p(vv)
            if (self.channels == ["vv", "vh", "vh/vv"] or
                (len(self.channels) == 3 and "vv" in self.channels and "vh" in self.channels and "vh/vv" in self.channels)):
                if img.shape[0] >= 3:
                    img[2].copy_(img[1] - img[0])
                elif img.shape[0] == 2:
                    # 扩展为3通道
                    C, H, W = img.shape
                    out = torch.empty((3, H, W), dtype=img.dtype, device=img.device)
                    out[0].copy_(img[0])
                    out[1].copy_(img[1])
                    out[2].copy_(img[1] - img[0])
                    img = out
            if self.data_mean is not None and self.data_std is not None:
                means = _align_stats(self.data_mean, "mean")
                stds = _align_stats(self.data_std, "std")
                scaled = torchvision.transforms.Normalize(means, stds)(img)
                return means, stds, scaled
            else:
                ones = torch.ones((img.shape[0],), device=img.device, dtype=img.dtype)
                zeros = torch.zeros((img.shape[0],), device=img.device, dtype=img.dtype)
                return zeros, ones, img

        elif self.scale_input == "db":
            # 线性强度转分贝（dB）：10*log10(max(x, eps))，原地计算
            eps = 1e-7
            img = img.clamp_min_(eps)
            img.log10_().mul_(10.0)
            # 第三通道改为 dB 比值：vh_db - vv_db
            if (self.channels == ["vv", "vh", "vh/vv"] or
                (len(self.channels) == 3 and "vv" in self.channels and "vh" in self.channels and "vh/vv" in self.channels)):
                if img.shape[0] >= 3:
                    img[2].copy_(img[1] - img[0])
                elif img.shape[0] == 2:
                    C, H, W = img.shape
                    out = torch.empty((3, H, W), dtype=img.dtype, device=img.device)
                    out[0].copy_(img[0])
                    out[1].copy_(img[1])
                    out[2].copy_(img[1] - img[0])
                    img = out
            if self.data_mean is not None and self.data_std is not None:
                means = _align_stats(self.data_mean, "mean")
                stds = _align_stats(self.data_std, "std")
                scaled = torchvision.transforms.Normalize(means, stds)(img)
                return means, stds, scaled
            else:
                ones = torch.ones((img.shape[0],), device=img.device, dtype=img.dtype)
                zeros = torch.zeros((img.shape[0],), device=img.device, dtype=img.dtype)
                return zeros, ones, img

        else:
            # 未知模式：不做缩放
            ones = torch.ones((img.shape[0],), device=img.device, dtype=img.dtype)
            zeros = torch.zeros((img.shape[0],), device=img.device, dtype=img.dtype)
            return zeros, ones, img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取单个样本（洪水后图像 + 可选 DEM）"""
        sample = self.samples[index]

        # 解析样本目录（兼容 pickle 中 path 相对 data/ 的情况）
        base_dir = self._resolve_sample_dir(sample["path"])
        files = os.listdir(base_dir)
        activation = sample["activation"]

        # 初始化图像变量（仅洪水后图像）
        flood_vv, flood_vh = None, None
        dem = None
        mask, valid_mask = None, None

        for file in files:
            if file.endswith('.xml'):
                continue
            current_path = os.path.join(base_dir, file)
            if file.startswith("MK0_MLU"):
                mask = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
            elif file.startswith("MK0_MNA"):
                valid_mask = cv.imread(current_path, cv.IMREAD_ANYDEPTH)
            elif file.startswith("MS1_IVV"):
                flood_vv = cv.imread(current_path, cv.IMREAD_ANYDEPTH).astype(np.float32)
            elif file.startswith("MS1_IVH"):
                flood_vh = cv.imread(current_path, cv.IMREAD_ANYDEPTH).astype(np.float32)
            elif file.startswith("MK0_DEM"):
                # DEM 原始高度值（单位与官方保持一致，通常为米）
                dem = cv.imread(current_path, cv.IMREAD_ANYDEPTH).astype(np.float32)

        if flood_vv is None or flood_vh is None:
            raise ValueError(f"缺少洪水后影像: {base_dir}")
        if self.dem and dem is None:
            raise ValueError(f"KuroSiwoDataset: dem=True 但样本缺少 MK0_DEM*.tif: {base_dir}")

        if mask is None:
            mask = np.zeros((224, 224), dtype=np.uint8)
        if valid_mask is None:
            valid_mask = np.ones((224, 224), dtype=np.uint8)

        # 转换为torch张量
        mask = torch.from_numpy(mask).long()
        valid_mask = torch.from_numpy(valid_mask)

        # 将标签二值化并设置忽略像素：
        # 假定 MK0_MLU 语义：1=洪水，2=永久水，3=无效；其余为非水
        # - 二值化：1/2 -> 1（水），其余 -> 0（非水）
        # - 忽略：valid_mask!=1 或 raw==3 -> -1
        raw_mask = mask.clone()
        binary = ((raw_mask == 1) | (raw_mask == 2)).long()
        ignore = (valid_mask != 1) | (raw_mask == 3)
        binary[ignore] = -1
        mask = binary

        # 连接洪水后图像通道（仅 SAR，后续再与 DEM 融合）
        flood = self._concat_channels(flood_vv, flood_vh)

        # 归一化前的诊断：原始通道最小/最大与夹紧比例（0 与 clamp 上限）
        raw_min = flood.amin(dim=(1, 2))
        raw_max = flood.amax(dim=(1, 2))
        if self.clamp_input is not None:
            eps = 1e-6
            sat_hi = (flood >= (self.clamp_input - eps)).float().mean(dim=(1, 2))
            sat_lo = (flood <= (0.0 + eps)).float().mean(dim=(1, 2))
        else:
            sat_hi = torch.zeros(flood.shape[0], dtype=torch.float32)
            sat_lo = torch.zeros(flood.shape[0], dtype=torch.float32)

        if self.scale_input is not None:
            valid_mask_tensor = valid_mask == 1
            _, _, flood = self._scale_image(flood, valid_mask_tensor, "flood", activation)

        # DEM 早期融合（追加为第3通道），支持 z-score / none 两种模式
        if self.dem and dem is not None:
            dem_t = torch.from_numpy(dem).float()  # [H,W]
            # 使用 valid_mask 屏蔽无效像素
            dem_valid = (valid_mask == 1)
            if self.dem_scale_mode == "zscore":
                mean = float(self.dem_mean)
                std = float(self.dem_std) if abs(self.dem_std) > 1e-6 else 1.0
                dem_t = (dem_t - mean) / std
            elif self.dem_scale_mode == "none":
                # 不做归一化，只在无效区域用全局均值填补，避免极端值
                pass
            else:
                raise ValueError(f"Unsupported dem_scale_mode: {self.dem_scale_mode}")
            # 在无效区域填充为0（与z-score后均值附近一致）
            dem_t[~dem_valid] = 0.0
            dem_t = torch.nan_to_num(dem_t, 0.0)
            # 追加到 SAR 通道后，形成 [vv, vh, dem]
            image = torch.cat([flood, dem_t.unsqueeze(0)], dim=0)
        else:
            image = flood

        output = {
            "image": image,
            "mask": mask,
            "activation": activation,
            "clz": sample["clz"],
            "path": sample["path"],
            # 诊断信息（按批在 DataModule 中汇总打印）
            "sat_hi": sat_hi,  # 每通道高端夹紧占比
            "sat_lo": sat_lo,  # 每通道低端夹紧占比
            "raw_min": raw_min,
            "raw_max": raw_max,
        }
        # 不再输出 dem

        if self.transforms is not None:
            output = self.transforms(output)

        return output
