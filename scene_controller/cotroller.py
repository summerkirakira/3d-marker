from abc import ABC, abstractmethod
from PIL import Image

class SceneController(ABC):

    def __init__(self, width=800, height=800):
        self._width = width
        self._height = height

    @abstractmethod
    def set_camera_pose(self, pose):
        pass

    @abstractmethod
    def get_left_view(self) -> Image:
        pass

    @abstractmethod
    def get_right_view(self) -> Image:
        pass

    @abstractmethod
    def get_front_view(self) -> Image:
        pass

    @abstractmethod
    def get_back_view(self) -> Image:
        pass
    
    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height
    
    @abstractmethod
    def look_at(self, object_id):
        pass
    
    
    
    
