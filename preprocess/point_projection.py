import numpy as np
from plyfile import PlyData
import json
from pathlib import Path

# 读取PLY文件
ply_path = r"D:\Dataset\ScanNet\ScanNetRaw\scene0000_00\scene0000_00_vh_clean.ply"
plydata = PlyData.read(ply_path)

# 获取顶点数据
vertex_data = plydata['vertex']

# 获取点坐标
points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

# 获取分割json文件
seg_path = r"D:\Dataset\ScanNet\ScanNetRaw\scene0000_00\scene0000_00_vh_clean.segs.json"
with open(seg_path, 'r') as f:
    seg_data = json.load(f)


# 获取segIndices字段
seg_indices = seg_data['segIndices']

# 根据seg_indices聚合点云
points_by_seg = {}
for i, seg_index in enumerate(seg_indices):
    if seg_index not in points_by_seg:
        points_by_seg[seg_index] = []
    points_by_seg[seg_index].append(points[i])

# 打印每个seg_index对应的点云数量
for seg_index, points_list in points_by_seg.items():
    print(f"seg_index: {seg_index}, 点云数量: {len(points_list)}")


# 获取seg 聚合json
seg_path = r"D:\Dataset\ScanNet\ScanNetRaw\scene0000_00\scene0000_00_vh_clean.aggregation.json"
with open(seg_path, 'r') as f:
    seg_data = json.load(f)


    # 获取所有segGroups
    seg_groups = seg_data['segGroups']
    
    # 用于存储每个objectId对应的点云
    points_by_obj = {}
    
# 遍历每个segGroup
for group in seg_groups:
    obj_id = group['objectId'] 
    segments = group['segments']
    label = group['label']
    
    # 如果这个objectId还没有对应的点云列表，创建一个
    if obj_id not in points_by_obj:
        points_by_obj[obj_id] = {
            'label': label,
            'points': []
        }
        
    # 收集这个组内所有segments对应的点
    for seg_id in segments:
        if seg_id in points_by_seg:
            points_by_obj[obj_id]['points'].extend(points_by_seg[seg_id])


import numpy as np

folder_path = Path(r"D:\Dataset\ScanNet\ScanNetRaw\scene0000_00\images")

img_path = folder_path / "color"

pose_path = folder_path / "pose"

intr_path = folder_path / "intrinsic" / "intrinsic_color.txt"

pose_path = folder_path / "pose"

def read_camera_matrix(intr_path):
    """
    读取相机内参矩阵文件并返回矩阵数据
    Args:
        intr_path: 内参矩阵文件路径
    Returns:
        intr_matrix: 内参矩阵数据列表
    """
    with open(intr_path, "r") as f:
        intr = f.readlines()

        # 将内参矩阵数据转换为浮点数列表
        intr_matrix = []
        for line in intr:
            # 分割每行数据并转换为浮点数
            row = [float(x) for x in line.strip().split()]
            intr_matrix.append(row)
        
        return intr_matrix

intr_matrix = read_camera_matrix(intr_path)

width = intr_matrix[0][2]
height = intr_matrix[1][2]
print(width, height)


