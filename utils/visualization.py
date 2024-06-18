import matplotlib.pyplot as plt

def show_image(image):
    plt.imshow(image)
    plt.show()

def plot_lidar(lidar_data):
    plt.scatter(lidar_data[:, 0], lidar_data[:, 1], s=1)
    plt.show()

def plot_radar(radar_data):
    plt.scatter(radar_data[:, 0], radar_data[:, 1], s=1, c=radar_data[:, 3])
    plt.colorbar()
    plt.show()