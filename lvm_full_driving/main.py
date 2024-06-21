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
from utils.data_processing import save_image, log_control_command, ensure_dir
from utils.visualization import show_image, plot_lidar, plot_radar
import random
from dotenv import load_dotenv
import logging
import signal
import psutil
import GPUtil

load_dotenv()  # Load environment variables from .env file

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Handle signals for graceful shutdown
def signal_handler(sig, frame):
    logging.info('Received signal to terminate. Cleaning up...')
    cleanup()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def log_resource_usage():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    gpu_load = gpus[0].load * 100 if gpus else "N/A"
    logging.info(f"CPU: {cpu_percent}%, Memory: {memory_percent}%, GPU: {gpu_load}%")

def stitch_images(images):
    """Stitches a list of images into a single panoramic image using OpenCV."""
    images_cv = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in images]
    
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images_cv)
    
    if status != cv2.Stitcher_OK:
        raise Exception("Error in stitching images, status code: {}".format(status))
    
    stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(stitched_image)

def lidar_to_image(lidar_data, vis, image_size=(600, 800)):
    """Converts LiDAR point cloud data to a 2D image using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_data[:, :3])
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)
    image = cv2.resize(image, image_size)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def radar_to_image(radar_data, fig, ax, image_size=(600, 800)):
    """Converts radar data to a 2D image."""
    ax.clear()
    scatter = ax.scatter(radar_data[:, 0], radar_data[:, 1], c=radar_data[:, 3], cmap='viridis')
    plt.colorbar(scatter)
    ax.set_xlim([-50, 50])
    ax.set_ylim([0, 100])
    ax.set_aspect('equal', adjustable='box')
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return Image.fromarray(cv2.resize(image, image_size))

def cleanup():
    """Cleanup all actors and sensors."""
    try:
        for camera in cameras:
            camera.sensor.destroy()
        lidar.sensor.destroy()
        radar.sensor.destroy()
        vehicle.destroy()
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

def update_spectator_position(vehicle, spectator):
    """Update the spectator's position to follow the vehicle."""
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

# Initialize CARLA client and world
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

current_settings = world.get_settings()

# Set synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True 
settings.fixed_delta_seconds = 0.05  
world.apply_settings(settings)

# Initialize vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
spawn_points = world.get_map().get_spawn_points()
vehicle_bp.set_attribute('role_name', 'hero')
vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))

# Initialize the spectator
spectator = world.get_spectator()

# Initialize sensors
camera_transforms = [
    carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0, yaw=0, roll=0)),     # Front camera
    carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(pitch=0, yaw=180, roll=0)),  # Rear camera
    carla.Transform(carla.Location(x=0.0, y=-1.5, z=2.4), carla.Rotation(pitch=0, yaw=-90, roll=0)),  # Left side camera
    carla.Transform(carla.Location(x=0.0, y=1.5, z=2.4), carla.Rotation(pitch=0, yaw=90, roll=0)),   # Right side camera
    carla.Transform(carla.Location(x=1.5, y=-0.75, z=2.4), carla.Rotation(pitch=0, yaw=-45, roll=0)),  # Front-left camera
    carla.Transform(carla.Location(x=1.5, y=0.75, z=2.4), carla.Rotation(pitch=0, yaw=45, roll=0)),   # Front-right camera
    carla.Transform(carla.Location(x=-1.5, y=-0.75, z=2.4), carla.Rotation(pitch=0, yaw=-135, roll=0)),  # Rear-left camera
    carla.Transform(carla.Location(x=-1.5, y=0.75, z=2.4), carla.Rotation(pitch=0, yaw=135, roll=0)),    # Rear-right camera
    carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0, yaw=10, roll=0)),     # Front additional camera
    carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(pitch=0, yaw=170, roll=0))   # Rear additional camera
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

# Setup logging paths
log_dir = 'data/logs'
ensure_dir(log_dir)
image_dir = os.path.join(log_dir, 'images')
ensure_dir(image_dir)
command_log_path = os.path.join(log_dir, 'control_commands.txt')

start_time = time.time()
duration = 5 * 60  # 5 minutes

try:
    fig, ax = plt.subplots()  # Create figure once
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    while time.time() - start_time < duration:
        logging.info("Collecting sensor data...")
        log_resource_usage()
        
        # Tick the world to advance the simulation
        world.tick()

        # Update the spectator's position
        update_spectator_position(vehicle, spectator)

        # Get sensor data
        images = [camera.get_image() for camera in cameras]
        lidar_data = lidar.get_point_cloud()
        radar_data = radar.get_points()
        gps_data = gps.get_location()

        vehicle_velocity = vehicle.get_velocity()
        vehicle_rotation = vehicle.get_transform().rotation
        vehicle_acceleration = vehicle.get_acceleration()
        vehicle_angular_velocity = vehicle.get_angular_velocity()
        
        vehicle_stats = {
            'speed': np.linalg.norm([vehicle_velocity.x, vehicle_velocity.y, vehicle_velocity.z]),
            'location': vehicle.get_transform().location,
            'rotation': (vehicle_rotation.pitch, vehicle_rotation.yaw, vehicle_rotation.roll),
            'acceleration': (vehicle_acceleration.x, vehicle_acceleration.y, vehicle_acceleration.z),
            'angular_velocity': (vehicle_angular_velocity.x, vehicle_angular_velocity.y, vehicle_angular_velocity.z)
        }

        if all(image is not None for image in images) and lidar_data is not None and radar_data is not None and gps_data is not None:
            stitched_image = stitch_images(images)
            
            # Convert LiDAR and radar data to images
            lidar_image = lidar_to_image(lidar_data, vis=vis)
            radar_image = radar_to_image(radar_data, fig=fig, ax=ax)
            
            # Get control command from GPT-4V
            try:
                control_command = get_control_from_gpt4([stitched_image, lidar_image, radar_image], vehicle_stats)
            except Exception as e:
                logging.error(f"Error getting control command from GPT-4V: {e}")
                continue

            # Save images and log the decision
            timestamp = int(time.time() * 1000)
            save_image(stitched_image, os.path.join(image_dir, f'stitched_{timestamp}.jpg'))
            save_image(lidar_image, os.path.join(image_dir, f'lidar_{timestamp}.jpg'))
            save_image(radar_image, os.path.join(image_dir, f'radar_{timestamp}.jpg'))
            log_control_command(control_command, command_log_path)

            # Execute the control command
            try:
                exec(control_command)
                logging.info(f"Executed control command: {control_command}")
            except Exception as e:
                logging.error(f"Error executing control command: {e}")

        else:
            logging.warning("Waiting for all sensor data to be available...")

        # Small delay to prevent busy-waiting
        time.sleep(0.01)

except Exception as e:
    logging.error(f"An error occurred: {e}")
finally:
    cleanup()
    plt.close(fig)  # Close the matplotlib figure
    vis.destroy_window()  # Destroy the Open3D visualizer
    logging.info("Cleanup completed.")

# Restore original settings
world.apply_settings(current_settings)