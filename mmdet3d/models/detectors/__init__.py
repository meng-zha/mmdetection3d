# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2020-12-28 10:56:28
# @Last Modified by:   Your name
# @Last Modified time: 2020-12-28 10:56:47
from .base import Base3DDetector
from .centerpoint import CenterPoint
from .dynamic_voxelnet import DynamicVoxelNet
from .h3dnet import H3DNet
from .mvx_faster_rcnn import DynamicMVXFasterRCNN, MVXFasterRCNN
from .mvx_two_stage import MVXTwoStageDetector
from .parta2 import PartA2
from .ssd3dnet import SSD3DNet
from .votenet import VoteNet
from .voxelnet import VoxelNet
from .r3dnet import R3DNet

__all__ = [
    'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
    'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
    'CenterPoint', 'SSD3DNet','R3DNet'
]
