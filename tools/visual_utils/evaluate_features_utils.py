from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap


def get_bev_map_waymo(points, output_path):
    #points = points[:, 1:].cpu().numpy()  # Remove batch_idx, convert from tensor to numpy array

    # if points is tensor, convert to numpy
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    # Define the bounds and resolution of the BEV image
    x_range = (-75.2, 75.2)
    y_range = (-75.2, 75.2)
    z_range = (-2, 4)
    resolution = 0.1

    # Filter points within the bounds
    mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & \
           (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) & \
           (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    points = points[mask]

    # Convert points to BEV coordinates
    x_img = ((points[:, 0] - x_range[0]) / resolution).astype(np.int32)
    y_img = ((points[:, 1] - y_range[0]) / resolution).astype(np.int32)
    x_img = np.clip(x_img, 0, 1504 - 1)
    y_img = np.clip(y_img, 0, 1504 - 1)

    # Normalize intensity values
    intensity = points[:, 3]  # Assuming the fourth column is intensity
    intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())

    # Create BEV map, initialized to white
    lidar_bev_data = np.full((1504, 1504, 3), 255, dtype=np.uint8)  # Initialize as white

    # Map intensity to colors
    cmap = cm.get_cmap('viridis')  # Choose any colormap
    colors = cmap(intensity_normalized)[:, :3]  # Get RGB values

    for i in range(len(x_img)):
        lidar_bev_data[y_img[i], x_img[i]] = (colors[i] * 255).astype(np.uint8)  # Scale colors to [0, 255]

    plt.figure(figsize=(10, 10))
    plt.imshow(lidar_bev_data, aspect='auto')
    plt.axis('off')  
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


def get_bev_map_waymo_2(points, output_path):
     points = points[:, [1, 2, 0]]
 
     # Define the bounds and resolution of the BEV image
     x_range = (0, 1504)
     y_range = (0, 1504)
     z_range = (0, 41)
 
     # Filter points within the bounds
     mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & \
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) & \
            (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
     points = points[mask]
 
     # Convert points to BEV coordinates
     x_img = ((points[:, 0] - x_range[0])).astype(np.int32)
     y_img = ((points[:, 1] - y_range[0])).astype(np.int32)
     x_img = np.clip(x_img, 0, 1504 - 1)
     y_img = np.clip(y_img, 0, 1504 - 1)
 
     # Create BEV map, initialized to white
     lidar_bev_data = np.full((1504, 1504, 3), 255, dtype=np.uint8)  # Initialize as white
 
     for i in range(len(x_img)):
         lidar_bev_data[y_img[i], x_img[i]] = [0,0,0] 
     plt.figure(figsize=(10, 10))
     plt.imshow(lidar_bev_data, aspect='auto')
     plt.axis('off')  
     plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    
def get_bev_map_waymo_downsampled(points, output_path):
     points = points[:, [1, 2, 0]]
 
     # Define the bounds and resolution of the BEV image
     x_range = (0, 188)
     y_range = (0, 188)
     z_range = (0, 5)
 
     # Filter points within the bounds
     mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & \
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) & \
            (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
     points = points[mask]
 
     # Convert points to BEV coordinates
     x_img = ((points[:, 0] - x_range[0])).astype(np.int32)
     y_img = ((points[:, 1] - y_range[0])).astype(np.int32)
     x_img = np.clip(x_img, 0, 188 - 1)
     y_img = np.clip(y_img, 0, 188 - 1)
 
     # Create BEV map, initialized to white
     lidar_bev_data = np.full((188, 188, 3), 255, dtype=np.uint8)  # Initialize as white
 
     for i in range(len(x_img)):
         lidar_bev_data[y_img[i], x_img[i]] = [0,0,0] 
     plt.figure(figsize=(10, 10))
     plt.imshow(lidar_bev_data, aspect='auto')
     plt.axis('off')  
     plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

def viz_bev_mask(masked_nonempty, unmasked_nonempty, masked_empty, unmasked_empty, output_path):
    masked_nonempty = masked_nonempty.squeeze()
    unmasked_nonempty = unmasked_nonempty.squeeze()
    masked_empty = masked_empty.squeeze()
    unmasked_empty = unmasked_empty.squeeze()

    # 1) Create an integer label map: background=0, mask1=1, mask2=2, mask3=3, mask4=4
    label = torch.zeros_like(masked_nonempty, dtype=torch.int64)
    label[masked_nonempty] = 1
    label[unmasked_nonempty] = 2
    label[masked_empty] = 3
    label[unmasked_empty] = 4
    #print("debug empty", torch.sum(label==0), torch.sum(label==1), torch.sum(label==2), torch.sum(label==3), torch.sum(label==4))

    # 2) Convert to NumPy for plotting
    label_np = label.cpu().numpy()
    label_np = label_np.T

    # 3) Define a colormap: index 0 is background (transparent), 1–4 are your mask colors
    # User-specified colors in 0–255, converted to 0–1
    c1 = np.array([255, 255, 0]) / 255  # mask1, yellow
    c2 = np.array([255, 0, 0]) / 255  # mask2, red
    c3 = np.array([223, 210, 228]) / 255  # mask3, purple
    # Slightly darker version of c3 for mask4:
    c4 = c3 * 0.8

    # Build colormap: entry 0 is transparent background
    cmap = ListedColormap([
        (0, 0, 0, 0.0),          # 0: transparent
        (*c1, 0.6),              # 1: your first color
        (*c2, 0.6),              # 2: your second color
        (*c3, 0.6),              # 3: your third color
        (*c4, 0.6),              # 4: darker third color
    ])

    # 4) Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(label_np, cmap=cmap, vmin=0, vmax=len(cmap.colors)-1,interpolation='nearest')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def viz_bev_cosine_similarity(prediction, target, empty_token, output_path,  vmin=0.0, vmax=1.0):
    target = target.squeeze()
    prediction = prediction.squeeze()

    # cos_sim now holds cosine similarity with special handling for zero vectors
    cos_sim_bev = F.cosine_similarity(target, prediction, dim=0)
    print("debug cos sim", cos_sim_bev.shape)

    cos_sim_bev = cos_sim_bev.cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.matshow(cos_sim_bev,  vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('cosine similarity')
    plt.savefig(output_path)
    plt.close()


    empty_token = empty_token.squeeze()
    bev_x_shape = prediction.shape[1]
    bev_y_shape = prediction.shape[2]
    print("debug", prediction.shape, empty_token.shape, bev_x_shape, bev_y_shape, prediction.reshape(256, -1).reshape)
    cos_sim_empty = F.cosine_similarity(prediction, empty_token[:, None, None], dim=0).view(bev_x_shape, bev_y_shape)
    cos_sim_empty = cos_sim_empty.cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.matshow(cos_sim_empty,  vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('cosine similarity')
    plt.savefig(output_path.replace('.png', '_empty.png'))
    plt.close()
    

    return cos_sim_bev

def viz_histogram(data, output_path):
    plt.figure(figsize=(10, 10))
    plt.hist(data, bins=100)
    plt.title('histogram')
    plt.savefig(output_path)
    plt.close()