import carla
import numpy as np

class CameraSensor:
    def __init__(self, vehicle, transform):
        blueprint_library = vehicle.get_world().get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        self.camera = vehicle.get_world().spawn_actor(camera_bp, transform, attach_to=vehicle)
        self.image = None

    def listen(self):
        self.camera.listen(lambda image: self.process_image(image))

    def process_image(self, image):
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.image = array

    def get_image(self):
        return self.image

    def destroy(self):
        self.camera.destroy()