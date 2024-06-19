import os
import json
from PIL import Image

def save_image(image, path):
    """Saves a numpy array as an image."""
    image.save(path)

def log_control_command(control_command, log_path):
    """Log the control command to the specified log file."""
    with open(log_path, 'a') as log_file:
        log_file.write(control_command + '\n')

def ensure_dir(directory):
    """Ensure that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
