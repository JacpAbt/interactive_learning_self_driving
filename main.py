import os
import time
import carla
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
from models.gpt4v_integration import encode_image_base64, get_control_from_gpt4
from sensors.camera import CameraSensor
from sensors.lidar import LidarSensor
from sensors.radar import RadarSensor
from sensors.gps import GPSSensor
from utils.data_processing import log_decision, load_logs
from utils.visualization import show_image, plot_lidar, plot_radar

def stitch_images(images):
    """Stitches a list of images into a single panoramic image using OpenCV."""
    # Convert images to OpenCV format
    images_cv = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in images]
    
    # Initialize OpenCV stitcher
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images_cv)
    
    if status != cv2.Stitcher_OK:
        raise Exception("Error in stitching images, status code: {}".format(status))
    
    # Convert back to PIL Image
    stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
    stitched_image = Image.fromarray(stitched_image)
    return stitched_image

def lidar_to_image(lidar_data, image_size=(600, 800)):
    """Converts LiDAR point cloud data to a 2D image using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_data[:, :3])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    
    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)
    image = cv2.resize(image, image_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

def radar_to_image(radar_data, image_size=(600, 800)):
    """Converts radar data to a 2D image."""
    fig, ax = plt.subplots()
    scatter = ax.scatter(radar_data[:, 0], radar_data[:, 1], c=radar_data[:, 3], cmap='viridis')
    plt.colorbar(scatter)

    ax.set_xlim([-50, 50])
    ax.set_ylim([0, 100])
    ax.set_aspect('equal', adjustable='box')
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    image = cv2.resize(image, image_size)
    return Image.fromarray(image)

# Initialize CARLA client and world
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Initialize vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Initialize sensors
camera_transforms = [
    carla.Transform(carla.Location(x=1.5, z=2.4)),  # Front camera
    carla.Transform(carla.Location(x=-1.5, z=2.4)),  # Rear camera
    carla.Transform(carla.Location(x=0.0, y=-1.5, z=2.4)),  # Left side camera
    carla.Transform(carla.Location(x=0.0, y=1.5, z=2.4)),  # Right side camera
    carla.Transform(carla.Location(x=1.5, y=-0.75, z=2.4)),  # Front-left camera
    carla.Transform(carla.Location(x=1.5, y=0.75, z=2.4))   # Front-right camera
]
cameras = [CameraSensor(vehicle, transform) for transform in camera_transforms]
for camera in cameras:
    camera.listen()

lidar_transform = carla.Transform(carla.Location(z=2.0))
lidar = LidarSensor(vehicle, lidar_transform)
lidar.listen()

radar_transform = carla.Transform(carla.Location(x=2.8, z=1.0))
radar = RadarSensor(vehicle, radar_transform)
radar.listen()

gps = GPSSensor(vehicle)
gps.listen()

# Data collection loop
log_path = 'data/logs/drive_log.json'
os.makedirs(os.path.dirname(log_path), exist_ok=True)

start_time = time.time()
duration = 2 * 60  # Run for 2 minutes

# Fetch current world settings
current_settings = world.get_settings()

try:
    while time.time() - start_time < duration:
        # Get sensor data
        images = [camera.get_image() for camera in cameras]
        lidar_data = lidar.get_point_cloud()
        radar_data = radar.get_points()
        gps_data = gps.get_location()

        if all(image is not None for image in images) and lidar_data is not None and radar_data is not None and gps_data is not None:
            # Stitch images to create a 360-degree view
            stitched_image = stitch_images(images)
            
            # Convert LiDAR and radar data to images
            lidar_image = lidar_to_image(lidar_data)
            radar_image = radar_to_image(radar_data)
            
            # Freeze simulation
            world.apply_settings(carla.WorldSettings(
                synchronous_mode=current_settings.synchronous_mode,
                no_rendering_mode=True,
                fixed_delta_seconds=0.05,
                substepping=current_settings.substepping,
                max_substep_delta_time=current_settings.max_substep_delta_time,
                max_substeps=current_settings.max_substeps,
                max_culling_distance=current_settings.max_culling_distance,
                deterministic_ragdolls=current_settings.deterministic_ragdolls,
                tile_stream_distance=current_settings.tile_stream_distance,
                actor_active_distance=current_settings.actor_active_distance,
                spectator_as_ego=current_settings.spectator_as_ego
            ))
            
            # Get control command from GPT-4V
            control_command = get_control_from_gpt4([stitched_image, lidar_image, radar_image])

            # Log the decision
            log_decision(np.array(stitched_image), lidar_data, radar_data, gps_data, control_command, log_path)

            # Execute the control command
            exec(control_command)
            
            # Unfreeze simulation
            world.apply_settings(current_settings)

finally:
    for camera in cameras:
        camera.destroy()
    lidar.destroy()
    radar.destroy()
    vehicle.destroy()
