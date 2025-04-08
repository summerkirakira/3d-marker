from pathlib import Path
import json

folder_path = Path(r"D:\Dataset\ScanNet\ScanNetRaw\scene0000_00\images")

img_path = folder_path / "color"

pose_path = folder_path / "pose"

intr_path = folder_path / "intrinsic" / "intrinsic_color.txt"

def read_intrinsic_matrix(intr_path):
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

intr_matrix = read_intrinsic_matrix(intr_path)

width = intr_matrix[0][2]
height = intr_matrix[1][2]
print(width, height)

blender_dict = {
    "camera_model": "SIMPLE_PINHOLE",
    "orientation_override": "none",
    "camera_angle_x": 0.6911112070083618,
}

frames = []


# 获取所有图片并按照文件名数字排序
img_files = sorted(img_path.glob("*.jpg"), key=lambda x: int(x.stem))

# 每6张图片抽取一张
for i, img_file in enumerate(img_files):
    if i % 6 != 0:  # 跳过不是6的倍数的索引
        continue
        
    pose = pose_path / f"{img_file.stem}.txt"
    pose = read_intrinsic_matrix(pose)
    frames.append({
        "file_path": str(img_file.absolute()),
        "transform_matrix": pose,
        "w": width,
        "h": height,
    })

blender_dict["frames"] = frames

with open("transforms.json", "w") as f:
    json.dump(blender_dict, f)
    
