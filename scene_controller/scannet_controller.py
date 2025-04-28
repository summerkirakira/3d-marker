from scene_controller.cotroller import SceneController
import open3d as o3d
import numpy as np
import json
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.rendering import Open3DScene
from PIL import Image
from pathlib import Path
from typing import Optional


class ScanNetController(SceneController):
    def __init__(self, width=1024, height=750, scene_path: Optional[Path] = None):
        super().__init__(width, height)
        gui.Application.instance.initialize()
        self._window = gui.Application.instance.create_window("ScanNet", self.width, self.height)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self._window.renderer)
        self._window.add_child(self._scene)

        if scene_path is None:
            raise ValueError("scene_path is required")
        
        self._scene_path = scene_path
        self._scene_name = scene_path.stem
        self._label_json_path = scene_path / "objects_info.json"

        self._mesh = o3d.io.read_triangle_mesh(str(scene_path / "scene0000_00_vh_clean.ply"))
        self._mesh.compute_vertex_normals()

        self._material = self._load_material()
        self._scene.scene.add_geometry(self._scene_name, self._mesh, self._material)
        self.init_camera()
        self.add_3d_label_from_json(self._label_json_path)
        gui.Application.instance.run()
    
    def _load_material(self) -> rendering.MaterialRecord:
        # 创建材质
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultLit'  # 使用默认光照着色器
        mat.base_color = [0.9, 0.9, 0.9, 1.0]  # 深灰色基础颜色
        mat.base_metallic = 0.1  # 非金属
        mat.base_roughness = 0.5  # 中等粗糙度
        mat.point_size = 3.0  # 点大小
        mat.absorption_distance = 0.5
        mat.transmission = 0.5
        mat.thickness = 1.0
        return mat
    
    def init_camera(self):
        bounds = self._mesh.get_axis_aligned_bounding_box()
        self._scene.setup_camera(60, bounds, bounds.get_center())
        camera = self._scene.scene.camera
        intrinsics = np.array([[1000, 0, 512],  # fx, 0, cx
                          [0, 1000, 375],    # 0, fy, cy  
                          [0, 0, 1]])        # 0, 0, 1
        camera.set_projection(intrinsics, 1, 1000.0, 1024, 750)
    
    def add_3d_label(self, center, object_id):
        label = self._scene.add_3d_label(center, object_id)
        label.color = gui.Color(1, 0, 0)
        label.scale = 2
        return label
    
    def add_3d_label_from_json(self, json_path: Path):
        with open(json_path, 'r', encoding='utf-8') as f:
            objects_info = json.load(f)
        
        for obj_id, obj_data in objects_info.items():
            label_text = obj_data['label']
            if label_text in ['wall', 'floor', 'ceiling', 'object']:
                continue
            label = self.add_3d_label(obj_data['center'], obj_id)
            label.color = gui.Color(1, 0, 0)
            label.scale = 2







    def set_camera_pose(self, pose):
        pass


    def get_left_view(self) -> Image:
        pass

    def get_right_view(self) -> Image:
        pass

    def get_front_view(self) -> Image:
        pass

    def get_back_view(self) -> Image:
        pass

    def look_at(self, object_id):
        pass

