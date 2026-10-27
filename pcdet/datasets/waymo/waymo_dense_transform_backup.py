# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Modified for Occupancy Completion / Object Densification

import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
import SharedArray
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from functools import partial

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate

# --- HELPER FUNCTIONS FOR SVD & GEOMETRY ---
def get_box_corners_np(boxes):
    """
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    Returns:
        corners: (N, 8, 3)
    """
    def rotz(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    n = boxes.shape[0]
    l, w, h = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    
    # Canonical corners (N, 8, 3)
    x_corners = np.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], axis=1)
    y_corners = np.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], axis=1)
    z_corners = np.stack([-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2], axis=1)
    
    corners = np.stack([x_corners, y_corners, z_corners], axis=2)
    
    # Rotate and Translate
    for i in range(n):
        R = rotz(boxes[i, 6])
        corners[i] = np.dot(corners[i], R.T)
    
    corners += boxes[:, :3].reshape(n, 1, 3)
    return corners

def mask_points_in_box(points, box):
    """
    Args:
        points: (N, 3)
        box: (7,) [x, y, z, dx, dy, dz, heading]
    Returns:
        mask: (N,) boolean
    """
    # Transform points to box local coordinate system
    pts = points - box[:3]
    angle = -box[6]
    c = np.cos(angle)
    s = np.sin(angle)
    
    pts_x = pts[:, 0] * c - pts[:, 1] * s
    pts_y = pts[:, 0] * s + pts[:, 1] * c
    pts_z = pts[:, 2]
    
    mask = (np.abs(pts_x) <= box[3] / 2) & \
           (np.abs(pts_y) <= box[4] / 2) & \
           (np.abs(pts_z) <= box[5] / 2)
    return mask
# -------------------------------------------

class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        print("debug", self.root_path)
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.infos = []
        self.seq_name_to_infos = self.include_waymo_data(self.mode)

        self.use_shared_memory = self.dataset_cfg.get('USE_SHARED_MEMORY', False) and self.training
        if self.use_shared_memory:
            self.shared_memory_file_limit = self.dataset_cfg.get('SHARED_MEMORY_FILE_LIMIT', 0x7FFFFFFF)
            self.load_data_to_shared_memory()

        if self.dataset_cfg.get('USE_PREDBOX', False):
            self.pred_boxes_dict = self.load_pred_boxes_to_dict(
                pred_boxes_path=self.dataset_cfg.ROI_BOXES_PATH[self.mode]
            )
        else:
            self.pred_boxes_dict = {}

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.seq_name_to_infos = self.include_waymo_data(self.mode)

    def include_waymo_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Waymo dataset')
        waymo_infos = []
        seq_name_to_infos = {}

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

            seq_name_to_infos[infos[0]['point_cloud']['lidar_sequence']] = infos

        self.infos.extend(waymo_infos[:])
        if self.logger is not None:
            self.logger.info('Total skipped info %s' % num_skipped_infos)
            self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            if self.logger is not None:
                self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))
        
        return seq_name_to_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        if not self.dataset_cfg.get('DISABLE_NLZ_FLAG_ON_POINTS', False):
            points_all = points_all[NLZ_flag == -1]
        
        # Note: If running offline generation, you might want to SKIP tanh normalization here
        # to save raw coordinates, or keep it consistent.
        if self.dataset_cfg.get('POINTS_TANH_DIM', None) is None:
            points_all[:, 3] = np.tanh(points_all[:, 3])
        else:
            for dim_idx in self.dataset_cfg.POINTS_TANH_DIM:
                points_all[:, dim_idx] = np.tanh(points_all[:, dim_idx])
        return points_all

    # --- NEW METHOD: Frame-by-Frame Occupancy Completion ---
    def generate_densified_frame(self, index, pre_frames=3, post_frames=0):
        """
        Generates a densified point cloud for the frame at 'index' using SVD alignment
        on objects and ego-motion on background from neighboring frames.
        """
        info = self.infos[index]
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        
        # 1. Load Current Frame
        points_cur = self.get_lidar(sequence_name, sample_idx)
        pose_cur = info['pose'].reshape((4, 4))
        
        # Check Tracking Info
        has_tracking = 'annos' in info and 'obj_ids' in info['annos']
        if has_tracking:
            annos_cur = info['annos']
            gt_ids_cur = annos_cur['obj_ids']
            gt_boxes_cur = annos_cur['gt_boxes_lidar']
            gt_corners_cur = get_box_corners_np(gt_boxes_cur)
        
        # 2. Define Neighbors
        # We can look at PRE and POST frames if available in the sequence
        sequence_infos = self.seq_name_to_infos[sequence_name]
        max_idx = len(sequence_infos) - 1
        
        neighbor_indices = []
        # Pre frames
        for i in range(1, pre_frames + 1):
            if sample_idx - i >= 0: neighbor_indices.append(sample_idx - i)
        # Post frames
        for i in range(1, post_frames + 1):
            if sample_idx + i <= max_idx: neighbor_indices.append(sample_idx + i)
            
        points_to_merge = [points_cur]
        
        # 3. Iterate Neighbors
        for n_idx in neighbor_indices:
            # Load Neighbor Data
            points_pre = self.get_lidar(sequence_name, n_idx)
            pose_pre = sequence_infos[n_idx]['pose'].reshape((4, 4))
            
            if has_tracking:
                annos_pre = sequence_infos[n_idx]['annos']
                gt_ids_pre = annos_pre['obj_ids']
                gt_boxes_pre = annos_pre['gt_boxes_lidar']
                gt_corners_pre = get_box_corners_np(gt_boxes_pre)
                
                # Point Status: 0=Bg, 1=Obj, -1=Ghost
                point_status = np.zeros(points_pre.shape[0], dtype=np.int8)
                
                # Check Objects
                for k, obj_id in enumerate(gt_ids_pre):
                    # Identify points in PRE box
                    box_mask = mask_points_in_box(points_pre[:, :3], gt_boxes_pre[k])
                    if not np.any(box_mask): continue
                    
                    if obj_id not in gt_ids_cur:
                        # GHOST: Object gone in current frame
                        point_status[box_mask] = -1
                    else:
                        # OBJECT: Align via SVD
                        point_status[box_mask] = 1
                        
                        # Find match
                        cur_k = np.where(gt_ids_cur == obj_id)[0][0]
                        
                        # SVD Alignment
                        src_corners = gt_corners_pre[k]
                        dst_corners = gt_corners_cur[cur_k]
                        
                        c_src = np.mean(src_corners, axis=0)
                        c_dst = np.mean(dst_corners, axis=0)
                        
                        src_centered = src_corners - c_src
                        dst_centered = dst_corners - c_dst
                        
                        # Rotation matrix from src to dst
                        A = np.dot(dst_centered.T, src_centered)
                        U, _, Vt = np.linalg.svd(A, full_matrices=False)
                        R = np.dot(Vt.T, U.T)
                        
                        # Apply to points (Pre Local -> Cur Local)
                        pts_local = points_pre[box_mask, :3]
                        pts_aligned = np.dot(pts_local - c_src, R.T) + c_dst
                        points_pre[box_mask, :3] = pts_aligned
                
                # Transform Background (Status == 0)
                bg_mask = (point_status == 0)
                if np.any(bg_mask):
                    pts_bg = points_pre[bg_mask, :3]
                    expand_pts = np.concatenate([pts_bg, np.ones((pts_bg.shape[0], 1))], axis=-1)
                    # Global -> Cur
                    pts_global = np.dot(expand_pts, pose_pre.T)
                    pts_cur = np.dot(pts_global, np.linalg.inv(pose_cur.T))[:, :3]
                    points_pre[bg_mask, :3] = pts_cur
                    
                # Filter Ghosts
                points_pre = points_pre[point_status != -1]
                
            else:
                # Fallback: Simple Ego-Motion for all points
                expand_pts = np.concatenate([points_pre[:, :3], np.ones((points_pre.shape[0], 1))], axis=-1)
                pts_global = np.dot(expand_pts, pose_pre.T)
                pts_cur = np.dot(pts_global, np.linalg.inv(pose_cur.T))[:, :3]
                points_pre[:, :3] = pts_cur
            
            # Optional: Add time lag feature if desired
            # time_lag = 0.1 * (sample_idx - n_idx)
            # points_pre = np.hstack([points_pre, ...])
            
            points_to_merge.append(points_pre)

        # Merge
        points_final = np.concatenate(points_to_merge, axis=0).astype(np.float32)
        return points_final

    # -------------------------------------------------------------

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        # Standard implementation (Unchanged)
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        input_dict = {
            'sample_idx': sample_idx
        }
        if self.use_shared_memory and index < self.shared_memory_file_limit:
            sa_key = f'{sequence_name}___{sample_idx}'
            points = SharedArray.attach(f"shm://{sa_key}").copy()
        else:
            points = self.get_lidar(sequence_name, sample_idx)

        # NOTE: Standard sequence loading is preserved here if needed for other models
        if self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED:
             # Just pass standard points for now, or use original get_sequence_data if needed
             pass

        input_dict.update({
            'points': points,
            'frame_id': info['frame_id'],
        })

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.dataset_cfg.get('TRAIN_WITH_SPEED', False):
                assert gt_boxes_lidar.shape[-1] == 9
            else:
                gt_boxes_lidar = gt_boxes_lidar[:, 0:7]

            if self.training and self.dataset_cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN', False):
                mask = (annos['num_points_in_gt'] > 0)
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)
        return data_dict
    
    # ... (Keep other methods like load_pred_boxes_to_dict, load_data_to_shared_memory etc.) ...
    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if not sequence_file.exists():
            found_sequence_file = sequence_file
            for pre_text in ['training', 'validation', 'testing']:
                if not sequence_file.exists():
                    temp_sequence_file = Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                    if temp_sequence_file.exists():
                        found_sequence_file = temp_sequence_file
                        break
            if not found_sequence_file.exists():
                found_sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))
            if found_sequence_file.exists():
                sequence_file = found_sequence_file
        return sequence_file


# --- EXECUTION BLOCK FOR OFFLINE TRANSFORMATION ---
if __name__ == '__main__':
    import argparse
    from pcdet.config import cfg, cfg_from_yaml_file
    
    parser = argparse.ArgumentParser(description='Offline Waymo Occupancy Completion')
    parser.add_argument('--cfg_file', type=str, default='cfgs/dataset_configs/waymo_dataset.yaml')
    parser.add_argument('--save_dir', type=str, required=True, help='Output directory for dense point clouds')
    parser.add_argument('--pre_frames', type=int, default=70, help='Number of previous frames to aggregate')
    parser.add_argument('--post_frames', type=int, default=70, help='Number of future frames to aggregate')
    parser.add_argument('--idx_start', type=int, default=0, help='Starting index to aggregate')
    parser.add_argument('--idx_end', type=int, default=1000, help='Ending index to aggregate')


    args = parser.parse_args()
    
    # Load Config
    cfg_from_yaml_file(args.cfg_file, cfg)
    
    # Init Dataset (Training mode=True to load all annotations)
    dataset = WaymoDataset(
        dataset_cfg=cfg,
        class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
        training=True,
        root_path=Path('/'),
        logger=common_utils.create_logger()
    )
    
    print(f"Dataset Loaded: {len(dataset)} frames.")
    print(f"Generating dense clouds (Pre: {args.pre_frames}, Post: {args.post_frames})...")
    
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Iterate and Process
    for i in tqdm(range(args.idx_start, args.idx_end)):
        try:
            # 1. Generate Dense Frame
            dense_points = dataset.generate_densified_frame(
                i, 
                pre_frames=args.pre_frames, 
                post_frames=args.post_frames
            )
            
            # 2. Determine Output Path
            info = dataset.infos[i]
            seq_name = info['point_cloud']['lidar_sequence']
            sample_idx = info['point_cloud']['sample_idx']
            
            seq_dir = save_path / seq_name
            seq_dir.mkdir(exist_ok=True)
            
            out_file = seq_dir / ('%04d.npy' % sample_idx)
            
            # 3. Save
            np.save(str(out_file), dense_points)
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")

    print("Processing Complete.")