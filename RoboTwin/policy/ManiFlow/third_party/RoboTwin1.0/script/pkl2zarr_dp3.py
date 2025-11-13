import pdb, pickle, os
import numpy as np
import cv2
import open3d as o3d
from copy import deepcopy
import zarr, shutil
import argparse
from diffusion_policy_3d.env.robotwin.utils import *

def crop_point_cloud(pcd, pcd_crop_bbox = [[-0.6, -0.35, 0.7413],[0.6, 0.35, 2]], num_points=1024):
    # pcd: (N, 6) np.array
    min_bound = pcd_crop_bbox[0]
    max_bound = pcd_crop_bbox[1]
    pcd = pcd[(pcd[:, 0] > min_bound[0]) & (pcd[:, 0] < max_bound[0])]
    pcd = pcd[(pcd[:, 1] > min_bound[1]) & (pcd[:, 1] < max_bound[1])]
    pcd = pcd[(pcd[:, 2] > min_bound[2]) & (pcd[:, 2] < max_bound[2])]
    if pcd.shape[0] > num_points:
        # fps sampling
        _, indices = fps(pcd, num_points=num_points, use_cuda=False)
        pcd = pcd[indices].squeeze()
    return pcd

def downsample_pointcloud(pointcloud, downsample_factor=2):
    """
    Downsample pointcloud using interpolation with a downsample factor
    Args:
        pointcloud: numpy array of shape (H, W, 6) where last dim is (x,y,z,r,g,b)
        downsample_factor: int, factor to downsample by (e.g. 2 means half resolution)
    Returns:
        downsampled pointcloud of shape (H//factor, W//factor, 6)
    """
    H, W, _ = pointcloud.shape
    target_h = H // downsample_factor
    target_w = W // downsample_factor
    
    # Separately handle xyz and rgb to use different interpolation methods
    xyz = pointcloud[..., :3]
    rgb = pointcloud[..., 3:]
    
    # Use bilinear for xyz to maintain geometric structure
    xyz_down = cv2.resize(xyz, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Use nearest neighbor for rgb to avoid color bleeding
    rgb_down = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    return np.concatenate([xyz_down, rgb_down], axis=-1)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('task_name', type=str)
    parser.add_argument('head_camera_type', type=str)
    parser.add_argument('expert_data_num', type=int)

    args = parser.parse_args()
    
    visualize_pcd = False
    downsample_factor = 4

    task_name = args.task_name
    num = args.expert_data_num
    current_ep, num = 0, num
    head_camera_type = args.head_camera_type
    load_dir = f'data/{task_name}_{head_camera_type}_pkl'
    
    total_count = 0

    save_dir = f'../../ManiFlow/data/{task_name}.zarr'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    point_cloud_arrays, episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = [], [], [], [], []

    # load_cameras = ['head_camera', 'front_camera', 'left_camera', 'right_camera']
    load_cameras = ['head_camera']
    head_point_cloud_arrays, front_point_cloud_arrays, left_point_cloud_arrays, right_point_cloud_arrays = [], [], [], []
    head_img_arrays, front_img_arrays, left_img_arrays, right_img_arrays = [], [], [], []
    head_depth_arrays, front_depth_arrays, left_depth_arrays, right_depth_arrays = [], [], [], []
    
    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0
        point_cloud_sub_arrays = []
        state_sub_arrays = []
        action_sub_arrays = [] 
        joint_action_sub_arrays = []
        episode_ends_sub_arrays = []
        head_point_cloud_sub_arrays, front_point_cloud_sub_arrays, left_point_cloud_sub_arrays, right_point_cloud_sub_arrays = [], [], [], []
        head_img_sub_arrays, front_img_sub_arrays, left_img_sub_arrays, right_img_sub_arrays = [], [], [], []
        head_depth_sub_arrays, front_depth_sub_arrays, left_depth_sub_arrays, right_depth_sub_arrays = [], [], [], []
        
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)
            print(f"Processing episode {current_ep}, file {file_num}", end='\r')

            pcd = data['pointcloud'][:,:]
            action = data['endpose'] # (7x2,) x, y, z, roll, pitch, yaw, gripper

            head_point_cloud_sub_sub_arrays, front_point_cloud_sub_sub_arrays, left_point_cloud_sub_sub_arrays, right_point_cloud_sub_sub_arrays = [], [], [], []
            head_img_sub_sub_arrays, front_img_sub_sub_arrays, left_img_sub_sub_arrays, right_img_sub_sub_arrays = [], [], [], []
            head_depth_sub_sub_arrays, front_depth_sub_sub_arrays, left_depth_sub_sub_arrays, right_depth_sub_sub_arrays = [], [], [], []

            for camera in load_cameras:
                this_camera_intrinsic = data['observation'][camera]['intrinsic_cv']
                this_camera_cam2world_matrix = data['observation'][camera]['cam2world_gl']
                this_color = data['observation'][camera]['rgb'][..., :3] / 255.0
                this_depth = data['observation'][camera]['depth'][...] / 1000
                if current_ep == 0 and file_num == 0:
                    print(f"this color range: {this_color.min()}, {this_color.max()}")
                    print(f"this depth range: {this_depth.min()}, {this_depth.max()}")
                H, W, _ = this_color.shape
                this_pcd_coords, this_pcd_colors = tanslation_point_cloud(
                    this_depth, 
                    this_color, 
                    this_camera_intrinsic, 
                    this_camera_cam2world_matrix)

                this_pcd = np.concatenate([this_pcd_coords, this_pcd_colors], axis=-1)
                this_pcd = this_pcd.reshape(H, W, 6)
                this_pcd = downsample_pointcloud(this_pcd, downsample_factor=downsample_factor)
                if camera == 'head_camera':
                    head_point_cloud_sub_sub_arrays.append(this_pcd)
                    # head_img_sub_sub_arrays.append(this_color)
                    # head_depth_sub_sub_arrays.append(this_depth)
                elif camera == 'front_camera':
                    pass
                elif camera == 'left_camera':
                    pass
                elif camera == 'right_camera':
                    pass

            head_point_cloud_sub_sub_arrays = np.array(head_point_cloud_sub_sub_arrays)
            # head_img_sub_sub_arrays = np.array(head_img_sub_sub_arrays)
            # head_depth_sub_sub_arrays = np.array(head_depth_sub_sub_arrays)

            joint_action = data['joint_action']

            point_cloud_sub_arrays.append(pcd)
            state_sub_arrays.append(joint_action)
            action_sub_arrays.append(action)
            joint_action_sub_arrays.append(joint_action)

            head_point_cloud_sub_arrays.append(head_point_cloud_sub_sub_arrays)
            # front_point_cloud_sub_arrays.append(front_point_cloud_arrays)
            # left_point_cloud_sub_arrays.append(left_point_cloud_arrays)
            # right_point_cloud_sub_arrays.append(right_point_cloud_arrays)

            # head_img_sub_arrays.append(head_img_sub_sub_arrays)
            # front_img_sub_arrays.append(front_img_arrays)
            # left_img_sub_arrays.append(left_img_arrays)
            # right_img_sub_arrays.append(right_img_arrays)

            # head_depth_sub_arrays.append(head_depth_sub_sub_arrays)
            # front_depth_sub_arrays.append(front_depth_arrays)
            # left_depth_sub_arrays.append(left_depth_arrays)
            # right_depth_sub_arrays.append(right_depth_arrays)
        

            if visualize_pcd:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data['pointcloud'][..., :3])
                pcd.colors = o3d.utility.Vector3dVector(data['pointcloud'][..., 3:6])
                # o3d.visualization.draw_geometries([pcd])
                # save pcd
                os.makedirs('./multi_view_pcd', exist_ok=True)
                o3d.io.write_point_cloud(f'./multi_view_pcd/{current_ep}_{file_num}.ply', pcd)

                # save head camera pcd
                head_pcd = o3d.geometry.PointCloud()
                head_point_cloud_array = head_point_cloud_arrays[-1].reshape(-1, 6)
                head_point_cloud_array = crop_point_cloud(head_point_cloud_array)
                head_pcd.points = o3d.utility.Vector3dVector(head_point_cloud_array[..., :3])
                head_pcd.colors = o3d.utility.Vector3dVector(head_point_cloud_array[..., 3:6] / 255.0)
                os.makedirs('./head_pcd', exist_ok=True)
                o3d.io.write_point_cloud(f'./head_pcd/{current_ep}_{file_num}.ply', head_pcd)


            file_num += 1
            total_count += 1
            
        current_ep += 1
        episode_ends_arrays.append(deepcopy(total_count))
        point_cloud_arrays.extend(point_cloud_sub_arrays)
        action_arrays.extend(action_sub_arrays)
        state_arrays.extend(state_sub_arrays)
        joint_action_arrays.extend(joint_action_sub_arrays)

        head_point_cloud_arrays.extend(head_point_cloud_sub_arrays)
        # front_point_cloud_arrays.extend(front_point_cloud_sub_arrays)
        # left_point_cloud_arrays.extend(left_point_cloud_sub_arrays)
        # right_point_cloud_arrays.extend(right_point_cloud_sub_arrays)
        
        # head_img_arrays.extend(head_img_sub_arrays)
        # front_img_arrays.extend(front_img_sub_arrays)
        # left_img_arrays.extend(left_img_sub_arrays)
        # right_img_arrays.extend(right_img_sub_arrays)

        # head_depth_arrays.extend(head_depth_sub_arrays)
        # front_depth_arrays.extend(front_depth_sub_arrays)
        # left_depth_arrays.extend(left_depth_sub_arrays)
        # right_depth_arrays.extend(right_depth_sub_arrays)


    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    point_cloud_arrays = np.array(point_cloud_arrays)
    joint_action_arrays = np.array(joint_action_arrays)

    head_point_cloud_arrays = np.array(head_point_cloud_arrays).squeeze()
    # front_point_cloud_arrays = np.array(front_point_cloud_arrays)
    # left_point_cloud_arrays = np.array(left_point_cloud_arrays)
    # right_point_cloud_arrays = np.array(right_point_cloud_arrays)

    # head_img_arrays = np.array(head_img_arrays).squeeze()
    # front_img_arrays = np.array(front_img_arrays)
    # left_img_arrays = np.array(left_img_arrays)
    # right_img_arrays = np.array(right_img_arrays)

    # head_depth_arrays = np.array(head_depth_arrays).squeeze()
    # front_depth_arrays = np.array(front_depth_arrays)
    # left_depth_arrays = np.array(left_depth_arrays)
    # right_depth_arrays = np.array(right_depth_arrays)

    # head_point_cloud_arrays = np.moveaxis(head_point_cloud_arrays, -1, 1)  # NHWC -> NCHW
    # head_img_arrays = np.moveaxis(head_img_arrays, -1, 1)  # NHWC -> NCHW
    # head_depth_arrays = np.moveaxis(head_depth_arrays, -1, 1)  # NHWC -> NCHW

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])

    head_point_cloud_chunk_size = (100, head_point_cloud_arrays.shape[1], head_point_cloud_arrays.shape[2], head_point_cloud_arrays.shape[3])
    # front_point_cloud_chunk_size = (100, front_point_cloud_arrays.shape[1], front_point_cloud_arrays.shape[2], front_point_cloud_arrays.shape[3])
    # left_point_cloud_chunk_size = (100, left_point_cloud_arrays.shape[1], left_point_cloud_arrays.shape[2], left_point_cloud_arrays.shape[3])
    # right_point_cloud_chunk_size = (100, right_point_cloud_arrays.shape[1], right_point_cloud_arrays.shape[2], right_point_cloud_arrays.shape[3])

    # head_img_chunk_size = (100, head_img_arrays.shape[1], head_img_arrays.shape[2], head_img_arrays.shape[3])
    # front_img_chunk_size = (100, front_img_arrays.shape[1], front_img_arrays.shape[2], front_img_arrays.shape[3])
    # left_img_chunk_size = (100, left_img_arrays.shape[1], left_img_arrays.shape[2], left_img_arrays.shape[3])
    # right_img_chunk_size = (100, right_img_arrays.shape[1], right_img_arrays.shape[2], right_img_arrays.shape[3])

    # head_depth_chunk_size = (100, head_depth_arrays.shape[1], head_depth_arrays.shape[2])
    # front_depth_chunk_size = (100, front_depth_arrays.shape[1], front_depth_arrays.shape[2])
    # left_depth_chunk_size = (100, left_depth_arrays.shape[1], left_depth_arrays.shape[2])
    # right_depth_chunk_size = (100, right_depth_arrays.shape[1], right_depth_arrays.shape[2])

    zarr_data.create_dataset('head_point_cloud', data=head_point_cloud_arrays, chunks=head_point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('front_point_cloud', data=front_point_cloud_arrays, chunks=front_point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('left_point_cloud', data=left_point_cloud_arrays, chunks=left_point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('right_point_cloud', data=right_point_cloud_arrays, chunks=right_point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

    # zarr_data.create_dataset('head_img', data=head_img_arrays, chunks=head_img_chunk_size, overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('front_img', data=front_img_arrays, chunks=front_img_chunk_size, overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('left_img', data=left_img_arrays, chunks=left_img_chunk_size, overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('right_img', data=right_img_arrays, chunks=right_img_chunk_size, overwrite=True, compressor=compressor)

    # zarr_data.create_dataset('head_depth', data=head_depth_arrays, chunks=head_depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('front_depth', data=front_depth_arrays, chunks=front_depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('left_depth', data=left_depth_arrays, chunks=left_depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('right_depth', data=right_depth_arrays, chunks=right_depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('tcp_action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=joint_action_arrays, chunks=joint_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

if __name__ == '__main__':
    main()

# example usage:
# python script/pkl2zarr_dp3.py mug_hanging_hard D435 50
# python script/pkl2zarr_dp3.py pick_apple_messy D435 50