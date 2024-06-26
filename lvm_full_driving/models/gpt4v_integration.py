from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
def encode_image_base64(image):
    """Encodes a PIL image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def get_control_from_gpt4(images, vehicle_stats):
    """Get control command from GPT-4V using multiple images."""
    encoded_images = [encode_image_base64(image) for image in images]
    stats_text = f"Vehicle stats: Speed: {vehicle_stats['speed']}, Location: {vehicle_stats['location']}, Rotation: {vehicle_stats['rotation']}, Acceleration: {vehicle_stats['acceleration']}, Angular velocity: {vehicle_stats['angular_velocity']}."

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
                        You are an autonomous driving system built within the CARLA open source program.
                        Analyze the driving environment and provide the exact CARLA vehicle control command. 
                        You'll recieve as input a stitched image of cameras around the car, lidar, and radar, and the stats of the car in the moment of this photo.
                        The response should be a single line of Python code to control the vehicle, like 'vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))'. 
                        Do not write 'python' at the beginning, do not put the command between ' ', just the pure python code.
                        These are the stats of the car in the moment of this photo: {stats_text}
                    """
                },
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    }
                    for image_base64 in encoded_images
                ],
            ],
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )
    print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()
