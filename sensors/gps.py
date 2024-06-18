import carla

class GPSSensor:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.gnss = None
        self.latitude = None
        self.longitude = None

    def listen(self):
        self.gnss = self.vehicle.get_world().spawn_actor(
            self.vehicle.get_world().get_blueprint_library().find('sensor.other.gnss'),
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.gnss.listen(lambda data: self.process_gnss(data))

    def process_gnss(self, data):
        self.latitude = data.latitude
        self.longitude = data.longitude

    def get_location(self):
        return self.latitude, self.longitude

    def destroy(self):
        self.gnss.destroy()