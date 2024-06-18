import carla
import numpy as np

class RadarSensor:
    def __init__(self, vehicle, transform):
        blueprint_library = vehicle.get_world().get_blueprint_library()
        radar_bp = blueprint_library.find('sensor.other.radar')
        self.radar = vehicle.get_world().spawn_actor(radar_bp, transform, attach_to=vehicle)
        self.points = None

    def listen(self):
        self.radar.listen(lambda data: self.process_radar(data))

    def process_radar(self, radar_data):
        data = np.frombuffer(radar_data.raw_data, dtype=np.float32)
        points = np.reshape(data, (-1, 4))
        self.points = points

    def get_points(self):
        return self.points

    def destroy(self):
        self.radar.destroy()