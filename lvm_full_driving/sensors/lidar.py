import carla
import numpy as np

class LidarSensor:
    def __init__(self, vehicle, transform):
        blueprint_library = vehicle.get_world().get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        self.lidar = vehicle.get_world().spawn_actor(lidar_bp, transform, attach_to=vehicle)
        self.point_cloud = None

    def listen(self):
        self.lidar.listen(lambda data: self.process_lidar(data))

    def process_lidar(self, point_cloud):
        data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        points = np.reshape(data, (int(data.shape[0] / 4), 4))
        self.point_cloud = points

    def get_point_cloud(self):
        return self.point_cloud

    def destroy(self):
        self.lidar.destroy()