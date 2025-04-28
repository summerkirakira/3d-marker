import numpy as np
from plyfile import PlyData
import json
from pathlib import Path
from typing import Tuple
from image_util import select_image_with_low_blur
import pickle
import shutil
IMAGE_INTERVAL = 100
IMAGE_CHECK_NUM = 20
IMAGE_SELECT_NUM = 8


def determine_occlusion_depth_sorting(points_3d, camera_pose):
    """
    使用深度排序法确定3D点的遮挡关系，考虑相机外参
    
    参数:
    points_3d: 形状为(N,3)的numpy数组，表示世界坐标系中的3D点
    camera_pose: 4x4相机外参矩阵，将世界坐标转换到相机坐标
    
    返回:
    sorted_indices: 按从远到近排序的点索引
    points_camera: 相机坐标系下的点坐标
    """
    # 将点转换为齐次坐标
    points_homogeneous = np.hstack((points_3d, np.ones((len(points_3d), 1))))
    
    # 将点从世界坐标系转换到相机坐标系
    points_camera = (camera_pose @ points_homogeneous.T).T
    points_camera = points_camera[:, :3]  # 去掉齐次坐标
    
    # 计算在相机坐标系下的距离
    distances = np.linalg.norm(points_camera, axis=1)
    
    # 按距离从远到近排序
    sorted_indices = np.argsort(distances)[::-1]
    
    return sorted_indices, points_camera


def visualize_depth_sorted_points_with_extrinsics(points_3d, camera_matrix, camera_pose, width, height, seg_to_obj, seg_indices):
    """
    包含外参的点投影可视化，使用z-buffer记录每个像素点的深度和索引
    
    参数:
    points_3d: 世界坐标系中的3D点
    camera_matrix: 相机内参矩阵
    camera_pose: 相机外参矩阵 (4x4)
    width, height: 图像尺寸
    """
    # 创建图像和z-buffer
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf)  # 初始化z-buffer为无穷大
    object_id_buffer = -1 * np.ones((height, width), dtype=np.int32)
    point_indices = np.full((height, width), -1, dtype=np.int32)  # 记录每个像素点对应的3D点索引
    
    # 获取内参
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    # 将点从世界坐标系转换到相机坐标系
    points_3d_homogeneous = np.hstack((points_3d, np.ones((len(points_3d), 1))))
    points_camera = (camera_pose @ points_3d_homogeneous.T).T
    points_camera = points_camera[:, :3]  # 去掉齐次坐标
    
    # 计算每个点到相机的距离
    distances = np.linalg.norm(points_camera, axis=1)
    sorted_indices = np.argsort(distances)[::-1]  # 从远到近排序
    
    # 按从远到近的顺序处理点
    for idx in sorted_indices:
        X_camera, Y_camera, Z_camera = points_camera[idx]
        
        # 忽略相机后方的点
        if Z_camera <= 0:
            continue
        
        # 投影到图像平面
        u = int(fx * X_camera / Z_camera + cx)
        v = int(fy * Y_camera / Z_camera + cy)
        
        # 检查点是否在图像范围内
        if 0 <= u < width and 0 <= v < height:
            # 定义矩形大小
            rect_size = 3  # 矩形边长的一半
            
            # 计算矩形边界
            left = max(0, u - rect_size)
            right = min(width - 1, u + rect_size)
            top = max(0, v - rect_size)
            bottom = min(height - 1, v + rect_size)
            
            # 遍历矩形内的所有点
            for ni in range(top, bottom + 1):
                for nj in range(left, right + 1):
                    # 如果当前点更近，则更新z-buffer和图像
                    if Z_camera < z_buffer[ni, nj]:
                        z_buffer[ni, nj] = Z_camera
                        point_indices[ni, nj] = idx
                        seg = seg_indices[idx]
                        if seg in seg_to_obj:
                            obj_id = seg_to_obj[seg]
                            # label = points_by_obj[obj_id]['label']
                            object_id_buffer[ni, nj] = obj_id
                            # if label == 'floor' or label == 'wall':
                            #     continue
                            # if label == 'tv':
                            #     color = np.array([0, 255, 0], dtype=np.uint8)
                            # elif label == 'curtain':
                            #     color = np.array([0, 0, 255], dtype=np.uint8)
                            # else:
                            # color = rgbs[idx]
                            # image[ni, nj] = color
    
    return image, z_buffer, point_indices, object_id_buffer


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
        
    # 将内参矩阵转换为numpy数组
    intr_matrix = np.array(intr_matrix)
    return intr_matrix


def greedy_selection(unique_point_indices, points, k=5):
    """
    使用贪心算法选择k张图片，使得这些图片覆盖最多的3D点
    
    Args:
        unique_point_indices: 每张图片可见的点的索引列表
        points: 所有3D点
        k: 要选择的图片数量
    Returns:
        selected_indices: 选中的图片索引
        coverage: 覆盖的点数量
    """
    n_images = len(unique_point_indices)
    selected_indices = []
    covered_points = set()
    
    for _ in range(k):
        best_idx = -1
        best_new_points = 0
        
        for i in range(n_images):
            if i in selected_indices:
                continue
                
            new_points = len(set(unique_point_indices[i]) - covered_points)
            if new_points > best_new_points:
                best_new_points = new_points
                best_idx = i
                
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        covered_points.update(unique_point_indices[best_idx])
        
    coverage = len(covered_points)
    
    return selected_indices, coverage, covered_points


def extract_data(scan_path: Path):
    scan_name = scan_path.name
    ply_path = scan_path / f"{scan_name}_vh_clean.ply"
    seg_path = scan_path / f"{scan_name}_vh_clean.segs.json"
    seg_aggregation_path = scan_path / f"{scan_name}_vh_clean.aggregation.json"
    image_path = scan_path / "images"
    intr_path = image_path / "intrinsic" / "intrinsic_color.txt"
    pose_path = image_path / "pose"
    img_path = image_path / "color"

    plydata = PlyData.read(ply_path)

    # 获取顶点数据
    vertex_data = plydata['vertex']
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    rgbs = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T

    seg_to_obj = {}
    seg_to_label = {}
    
    with open(seg_path, 'r') as f:
        seg_data = json.load(f)
 

    # 获取segIndices字段
    seg_indices = seg_data['segIndices']

    with open(seg_aggregation_path, 'r') as f:
        seg_aggregation_data = json.load(f)

    # 获取segGroups字段
    seg_groups = seg_aggregation_data['segGroups']

    for seg_group in seg_groups:
        for seg in seg_group["segments"]:
            seg_to_obj[seg] = seg_group["objectId"]
            seg_to_label[seg] = seg_group["label"]
    
    filtered_points = []

    for idx in range(len(seg_indices)):
        seg_id = seg_indices[idx]
        if seg_id not in seg_to_obj:
            continue
        if seg_to_label[seg_id] == "wall" or seg_to_label[seg_id] == "floor" or seg_to_label[seg_id] == "ceiling":
            continue
        filtered_points.append(points[idx])
    
    filtered_points = np.array(filtered_points)

    print(filtered_points.shape)

    intr_matrix = read_camera_matrix(intr_path)

    width = int(intr_matrix[0][2])
    height = int(intr_matrix[1][2])

    image_num = len(list(pose_path.glob("*.txt")))
    unique_point_indices = []

    max_item_coverage_dict = {}
    for i in range(len(seg_groups)):
        max_item_coverage_dict[i] = {
            "pixels": 0,
            "image_idx": -1
        }


    seg_2d_cache = {}
    used_image_list = []

    for i in range(image_num):
        if i % IMAGE_INTERVAL != 0:
            continue
        if i - IMAGE_CHECK_NUM < 0:
            start_idx = 0
            end_idx = IMAGE_CHECK_NUM
        elif i + IMAGE_CHECK_NUM > image_num:
            end_idx = image_num
            start_idx = image_num - IMAGE_CHECK_NUM
        else:
            start_idx = i - IMAGE_CHECK_NUM // 2
            end_idx = i + IMAGE_CHECK_NUM // 2
        check_image_list = [img_path / f"{i}.jpg" for i in range(start_idx, end_idx)]
        selected_image_path = select_image_with_low_blur(check_image_list)
        pose_file = pose_path / f"{selected_image_path.stem}.txt"
        used_image_list.append(selected_image_path)
        with open(pose_file, 'r') as f:
            camera_pose = read_camera_matrix(pose_file)
        camera_pose = np.linalg.inv(camera_pose)
        points_3d = points
        camera_matrix = intr_matrix[:3][:3]
        _, _, point_indices, object_id_buffer = visualize_depth_sorted_points_with_extrinsics(points_3d, camera_matrix, camera_pose, width, height, seg_to_obj, seg_indices)
        point_indices = point_indices[~np.isin(point_indices, filtered_points)]
        unique_point_indices.append(np.unique(point_indices))

        pixel_counter = np.zeros(len(seg_groups), dtype=np.int32)
        for i in range(object_id_buffer.shape[0]):
            for j in range(object_id_buffer.shape[1]):
                if object_id_buffer[i, j] != -1:
                    pixel_counter[object_id_buffer[i, j]] += 1
        
        for i in range(len(seg_groups)):
            if pixel_counter[i] > max_item_coverage_dict[i]["pixels"]:
                max_item_coverage_dict[i]["pixels"] = pixel_counter[i]
                max_item_coverage_dict[i]["image_idx"] = int(pose_file.stem)
            
        seg_2d_cache[int(pose_file.stem)] = object_id_buffer
        

        print(f"pose_file: {pose_file}, unique_point_indices: {len(np.unique(point_indices))}")

    selected_indices, coverage, covered_points = greedy_selection(unique_point_indices, points, k=IMAGE_SELECT_NUM)

    coverage_ratio = (coverage + len(filtered_points)) / (len(points))

    # covered_points = list(covered_points)
    # covered_points.extend(filtered_points)
    # covered_points = np.array(covered_points)
    
    print(f"选择了 {len(selected_indices)} 张图片")
    print(f"覆盖了 {coverage} 个点，覆盖率: {coverage_ratio:.2%}")


    used_image_foler = scan_path / "used_images"
    used_image_foler.mkdir(parents=True, exist_ok=True)

    used_item_image_list = [item["image_idx"] for item in max_item_coverage_dict.values()]

    new_used_image_list = []
    for i in range(len(used_image_list)):
        if i in selected_indices:
            new_used_image_list.append(used_image_list[i])

    new_seg_2d = {}
    cam_poses = []
    for key, value in seg_2d_cache.items():
        if key in used_item_image_list or key in new_used_image_list:
            new_seg_2d[key] = value
            shutil.copy(img_path / f"{key}.jpg", used_image_foler / f"{key}.jpg")
            cam_poses.append(read_camera_matrix(pose_path / f"{key}.txt"))
    scene_info = {
        "used_image_list": new_used_image_list,
        "max_item_coverage_dict": max_item_coverage_dict,
        "seg_2d": new_seg_2d,
        "cam_poses": cam_poses,
    }

    pickle.dump(scene_info, open(scan_path / f"{scan_name}_scene_info.pkl", "wb"))


if __name__ == "__main__":
    extract_data(Path(r"/Users/forever/Documents/Code/ScanNetMarker/scene0000_00"))

    
            