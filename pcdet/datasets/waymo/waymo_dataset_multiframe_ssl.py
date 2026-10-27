# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved.

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


class WaymoDataset_MultiFrame_SSL(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.infos = []
        self.seq_name_to_infos,  self.seq_name_to_infos_full = self.include_waymo_data(self.mode)

        self.use_shared_memory = self.dataset_cfg.get('USE_SHARED_MEMORY', False) and self.training
        if self.use_shared_memory:
            self.shared_memory_file_limit = self.dataset_cfg.get('SHARED_MEMORY_FILE_LIMIT', 0x7FFFFFFF)
            self.load_data_to_shared_memory()

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.seq_name_to_infos, self.seq_name_to_infos_full = self.include_waymo_data(self.mode)

    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []
        seq_name_to_infos = {}
        seq_name_to_infos_full = {}


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
                infos_full = infos.copy()

                # only load samples with valid frames with sample offset information
                infos = infos[self.dataset_cfg['SEQUENCE_CONFIG']['SAMPLE_OFFSET'][1]*self.dataset_cfg['SEQUENCE_CONFIG']['FRAME_SKIP']:self.dataset_cfg['SEQUENCE_CONFIG']['SAMPLE_OFFSET'][0]*self.dataset_cfg['SEQUENCE_CONFIG']['FRAME_SKIP']]

                #print("debug load seq infos", sequence_name, len(infos), info_path)
                waymo_infos.extend(infos)

            seq_name_to_infos[infos[0]['point_cloud']['lidar_sequence']] = infos
            seq_name_to_infos_full[infos_full[0]['point_cloud']['lidar_sequence']] = infos_full

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        sampled_waymo_infos = []
        sampled_ids = []
        for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
            sampled_waymo_infos.append(self.infos[k])
            sampled_ids.append(k)

        self.infos = sampled_waymo_infos
        self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))
          
                      
        use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED
        if not use_sequence_data:
            seq_name_to_infos = None 
        return seq_name_to_infos, seq_name_to_infos_full

    def load_data_to_shared_memory(self):
        self.logger.info(f'Loading training data to shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            points = self.get_lidar(sequence_name, sample_idx)
            common_utils.sa_create(f"shm://{sa_key}", points)

        dist.barrier()
        self.logger.info('Training data has been saved to shared memory')

    def clean_shared_memory(self):
        self.logger.info(f'Clean training data from shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if not os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            SharedArray.delete(f"shm://{sa_key}")

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('Training data has been deleted from shared memory')

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

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1, update_info_only=False):
        from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label, update_info_only=update_info_only
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        # process_single_sequence(sample_sequence_file_list[0])
        with multiprocessing.Pool(num_workers) as p:
            sequence_infos = list(tqdm(p.imap(process_single_sequence, sample_sequence_file_list),
                                       total=len(sample_sequence_file_list)))

        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        if not self.dataset_cfg.get('DISABLE_NLZ_FLAG_ON_POINTS', False):
            points_all = points_all[NLZ_flag == -1]
        if self.dataset_cfg.get('POINTS_TANH_DIM', None) is None:
            points_all[:, 3] = np.tanh(points_all[:, 3])
        else:
            for dim_idx in self.dataset_cfg.POINTS_TANH_DIM:
                points_all[:, dim_idx] = np.tanh(points_all[:, dim_idx])
        return points_all

    def get_sequence_data(self, info, points, sequence_name, sample_idx, sequence_cfg, load_pred_boxes=False):
        """
        Args:
            info:
            points:
            sequence_name:
            sample_idx:
            sequence_cfg:
        Returns:
        """

        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        pose_cur = info['pose'].reshape((4, 4))
        num_pts_cur = points.shape[0]
        
        sample_idx_list = np.clip(sample_idx + np.arange(sequence_cfg.SAMPLE_OFFSET[0]*sequence_cfg.FRAME_SKIP, (sequence_cfg.SAMPLE_OFFSET[1]+1)*sequence_cfg.FRAME_SKIP, sequence_cfg.FRAME_SKIP), 0, 0x7FFFFFFF)
        
        # timestamp dimension
        points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])
        
        points_all = []
        num_points = []
        pose_all = []


        sequence_info = self.seq_name_to_infos[sequence_name]
        sequence_info_full = self.seq_name_to_infos_full[sequence_name]
        for idx, sample_t in enumerate(sample_idx_list):
            if sample_t == sample_idx:
                points_all.append(points)
                num_points.append(num_pts_cur)
                pose_all.append(pose_cur)
            else:
                points_t = self.get_lidar(sequence_name, sample_t)
                points_t = remove_ego_points(points_t, 1.0) # fixing the bug at original openpcdet's codebase. Remove ego points before applying transformation, not after.
                pose_t = sequence_info_full[sample_t]['pose'].reshape((4, 4))
                expand_points_t= np.concatenate([points_t[:, :3], np.ones((points_t.shape[0], 1))], axis=-1)
                points_t_global = np.dot(expand_points_t, pose_t.T)[:, :3]
                expand_points_t_global = np.concatenate([points_t_global, np.ones((points_t_global.shape[0], 1))], axis=-1)
                points_t2cur = np.dot(expand_points_t_global, np.linalg.inv(pose_cur.T))[:, :3]
                points_t = np.concatenate([points_t2cur, points_t[:, 3:]], axis=-1)
                # add timestamp
                points_t = np.hstack([points_t, (sample_t-sample_idx) / sequence_cfg.FRAME_SKIP * np.ones((points_t.shape[0], 1)).astype(points_t.dtype)])  # one frame (0.1*sequence_cfg.FRAME_SKIP)s
                
                points_all.append(points_t)
                num_points.append(points_t.shape[0])
                pose_all.append(pose_t)

        #print("debug get seq points_all", len(points_all), [p.shape for p in points_all], points.shape)

        points = np.concatenate(points_all, axis=0).astype(np.float32)
        #print("debug get seq points shape 1", points.shape)
        
        num_points_all = np.array(num_points).astype(np.int32)
        poses = np.concatenate(pose_all, axis=0).astype(np.float32)

        points_shape = [point.shape for point in points_all]
        #print("debug get seq points shape 2", points_shape)

        return points, num_points_all, sample_idx_list, poses

        '''
        # timestamp dimension
        points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])
        points_pre_all = []
        num_points_pre = []

        pose_all = [pose_cur]
        

        sequence_info = self.seq_name_to_infos[sequence_name]
        sequence_info_full = self.seq_name_to_infos_full[sequence_name]
        for idx, sample_idx_pre in enumerate(sample_idx_list):
            if sample_idx_pre == sample_idx:
                continue

            points_pre = self.get_lidar(sequence_name, sample_idx_pre)
            points_pre = remove_ego_points(points_pre, 1.0) # fixing the bug at original openpcdet's codebase. Remove ego points before applying transformation, not after.
            pose_pre = sequence_info_full[sample_idx_pre]['pose'].reshape((4, 4))
            expand_points_pre = np.concatenate([points_pre[:, :3], np.ones((points_pre.shape[0], 1))], axis=-1)
            points_pre_global = np.dot(expand_points_pre, pose_pre.T)[:, :3]
            expand_points_pre_global = np.concatenate([points_pre_global, np.ones((points_pre_global.shape[0], 1))], axis=-1)
            points_pre2cur = np.dot(expand_points_pre_global, np.linalg.inv(pose_cur.T))[:, :3]
            points_pre = np.concatenate([points_pre2cur, points_pre[:, 3:]], axis=-1)
            # add timestamp
            points_pre = np.hstack([points_pre, (sample_idx_pre-sample_idx) / sequence_cfg.FRAME_SKIP * np.ones((points_pre.shape[0], 1)).astype(points_pre.dtype)])  # one frame (0.1*sequence_cfg.FRAME_SKIP)s
            
            points_pre_all.append(points_pre)
            num_points_pre.append(points_pre.shape[0])
            pose_all.append(pose_pre)


        print("debug get seq points_pre_all", len(points_pre_all), [p.shape for p in points_pre_all], points.shape)
        points = np.concatenate([points] + points_pre_all, axis=0).astype(np.float32)
        print("debug get seq points shape 1", points.shape)
        num_points_all = np.array([num_pts_cur] + num_points_pre).astype(np.int32)
        poses = np.concatenate(pose_all, axis=0).astype(np.float32)

        points_shape = [point.shape for point in points_pre_all]
        print("debug get seq points shape 2", points_shape)

        
        pred_boxes = pred_scores = pred_labels = None

        return points, num_points_all, sample_idx_list, poses
        '''

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
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

        #print("debug points before get_seq", points.shape)
        if self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED:
            points, num_points_all, sample_idx_list, poses = self.get_sequence_data(
                info, points, sequence_name, sample_idx, self.dataset_cfg.SEQUENCE_CONFIG,
                load_pred_boxes=self.dataset_cfg.get('USE_PREDBOX', False)
            )
            #print("debug points after get_seq", points.shape, np.unique(points[:, -1]))
            input_dict['poses'] = poses

        input_dict.update({
            'points': points,
            'frame_id': info['frame_id'],
        })

        data_dict = self.prepare_data_wo_box(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)
        # debug sample_idx_list
        data_dict['sample_idx_list'] = sample_idx_list
        data_dict['sequence_name'] = sequence_name
        return data_dict