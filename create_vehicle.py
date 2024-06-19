import carla
import random
import json

# Initialize CARLA client and world
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Initialize vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
spawn_points = world.get_map().get_spawn_points()
vehicle_bp.set_attribute('role_name', 'hero')
vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))

# Initialize sensors
camera_transforms = [
    carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0, yaw=0, roll=0)),     # Front camera
    carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(pitch=0, yaw=180, roll=0)),  # Rear camera
    carla.Transform(carla.Location(x=0.0, y=-1.5, z=2.4), carla.Rotation(pitch=0, yaw=-90, roll=0)),  # Left side camera
    carla.Transform(carla.Location(x=0.0, y=1.5, z=2.4), carla.Rotation(pitch=0, yaw=90, roll=0)),   # Right side camera
    carla.Transform(carla.Location(x=1.5, y=-0.75, z=2.4), carla.Rotation(pitch=0, yaw=-45, roll=0)),  # Front-left camera
    carla.Transform(carla.Location(x=1.5, y=0.75, z=2.4), carla.Rotation(pitch=0, yaw=45, roll=0)),   # Front-right camera
    carla.Transform(carla.Location(x=-1.5, y=-0.75, z=2.4), carla.Rotation(pitch=0, yaw=-135, roll=0)),  # Rear-left camera
    carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(pitch=0, yaw=135, roll=0))    # Rear-right camera
]
cameras = [world.spawn_actor(world.get_blueprint_library().find('sensor.camera.rgb'), transform, attach_to=vehicle) for transform in camera_transforms]

lidar_transform = carla.Transform(carla.Location(z=2.0))
lidar = world.spawn_actor(world.get_blueprint_library().find('sensor.lidar.ray_cast'), lidar_transform, attach_to=vehicle)

radar_transform = carla.Transform(carla.Location(x=2.8, z=1.0))
radar = world.spawn_actor(world.get_blueprint_library().find('sensor.other.radar'), radar_transform, attach_to=vehicle)

gps = world.spawn_actor(world.get_blueprint_library().find('sensor.other.gnss'), carla.Transform(), attach_to=vehicle)

# Store the actor IDs in a file
actor_ids = {
    'vehicle_id': vehicle.id,
    'camera_ids': [camera.id for camera in cameras],
    'lidar_id': lidar.id,
    'radar_id': radar.id,
    'gps_id': gps.id
}

with open('vehicle_sensors.json', 'w') as f:
    json.dump(actor_ids, f)

print("Vehicle and sensors created and stored.")
