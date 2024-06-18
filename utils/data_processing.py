import pandas as pd
import json
import numpy as np

def log_decision(image, lidar_data, radar_data, gps_data, decision, log_path):
    """Logs the decision with the corresponding sensor data."""
    log_entry = {
        'image': np.array(image).tolist(),
        'lidar_data': lidar_data.tolist(),
        'radar_data': radar_data.tolist(),
        'gps_data': gps_data,
        'decision': decision
    }
    with open(log_path, 'a') as log_file:
        log_file.write(json.dumps(log_entry) + '\n')

def load_logs(log_path):
    """Loads the log file into a DataFrame."""
    with open(log_path, 'r') as log_file:
        logs = [json.loads(line) for line in log_file]
    return pd.DataFrame(logs)