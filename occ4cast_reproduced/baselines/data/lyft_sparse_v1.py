# v1: lyft_sparse_timechannels_dataset.py
import os
import yaml
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class LyftDataset(Dataset):
    """
    Occ4D-Lyft dataset (sparse) — Option A: *time as channels*.
    - Input features: one channel per past frame (history+current), at each unique (z, y, x).
    - Sparse coords: unique (z, y, x) only (no duplicates).
    - Labels/invalid: dense (current+future) with shape [T_post, X, Y, Z].
    """

    def __init__(self, root_dir, split='train', p_pre=10, p_post=10, semantic=False):
        """
        Args:
            root_dir (str): Dataset root containing "voxel/" etc.
            split (str): 'train' | 'valid' | 'test'
            p_pre (int): number of frames before current (dataset adds +1 for current).
            p_post (int): number of frames after current (dataset adds +1 for current).
            semantic (bool): whether to return semantic label_fc from config.
        """
        self.root_dir = root_dir
        if split == 'train':
            self.split = list(range(120))
        elif split == 'valid':
            self.split = list(range(120, 150))
        elif split == 'test':
            self.split = list(range(150, 180))
        else:
            raise ValueError(f'Invalid split name: {split}')

        # +1 to include the current frame at t=0
        self.p_pre = p_pre + 1    # last p_pre frames (history) + current
        self.p_post = p_post + 1  # current + next p_post frames
        self.semantic = semantic

        # Grid (X, Y, Z) — kept consistent with your data
        self.X, self.Y, self.Z = 512, 256, 32

        with open("configs/lyft.yaml", 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self._load_data()

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _load_data(self):
        """Build a list of absolute stubs to each sample."""
        self.data_path = []
        voxel_root = os.path.join(self.root_dir, "voxel")

        for seq_id in self.split:
            seq_str = f"{seq_id:04d}"
            seq_dir = os.path.join(voxel_root, seq_str)
            if not os.path.isdir(seq_dir):
                continue

            for input_path in sorted(glob(os.path.join(seq_dir, "*_input.npy"))):
                name = os.path.basename(input_path)
                if not name.endswith("_input.npy"):
                    continue
                sample_id = name[:-10]  # strip "_input.npy"

                label_path   = os.path.join(seq_dir, f"{sample_id}_label.npy")
                invalid_path = os.path.join(seq_dir, f"{sample_id}_invalid.npy")

                if os.path.isfile(label_path) and os.path.isfile(invalid_path):
                    stub = os.path.join(os.path.abspath(seq_dir), sample_id)
                    self.data_path.append(stub)

    @staticmethod
    def _flat_ids_to_zyx(ids, X, Y, Z):
        """
        Convert flattened spatial IDs (over X*Y*Z with order [x, y, z]) into (z, y, x).
        Flattening order is assumed: id = x*(Y*Z) + y*Z + z
        """
        x = ids // (Y * Z)
        yz = ids % (Y * Z)
        y = yz // Z
        z = yz % Z
        return np.stack([z, y, x], axis=1).astype(np.int32)  # [N, 3] (z, y, x)

    # ---------------------------
    # Dataset API
    # ---------------------------

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        """
        Returns a dict with:
            voxel_features: (N, p_pre) float32, time-as-channels (history...current)
            voxel_coords:   (N, 3) int32  [z, y, x]
            labels:         (T_post, X, Y, Z) torch.Tensor
            invalid:        (T_post, X, Y, Z) torch.BoolTensor
        """
        stub = self.data_path[idx]
        X, Y, Z = self.X, self.Y, self.Z

        # --- LABELS (dense, assumed already un-packed) ---
        data_label = np.load(stub + "_label.npy")  # shape e.g. [11, X, Y, Z]
        # take current + future
        label = data_label[:self.p_post]  # (T_post, X, Y, Z)

        # --- INPUT (bit-packed), take last p_pre frames (history + current) ---
        data_input_arr = np.load(stub + "_input.npy")
        data_input_bits = np.unpackbits(data_input_arr)
        # Infer total time length dynamically
        T_total = int(data_input_bits.size // (X * Y * Z))
        data_input = data_input_bits.reshape(T_total, X, Y, Z)
        assert self.p_pre <= T_total, (
            f"Requested p_pre={self.p_pre} but only {T_total} frames available in INPUT."
        )
        input_hist = data_input[-self.p_pre:]  # shape (p_pre, X, Y, Z), uint8

        # --- INVALID (bit-packed), take current + future frames ---
        data_invalid_arr = np.load(stub + "_invalid.npy")
        data_invalid_bits = np.unpackbits(data_invalid_arr)
        T_total_inv = int(data_invalid_bits.size // (X * Y * Z))
        data_invalid = data_invalid_bits.reshape(T_total_inv, X, Y, Z)
        assert self.p_post <= T_total_inv, (
            f"Requested p_post={self.p_post} but only {T_total_inv} frames in INVALID."
        )
        invalid = data_invalid[:self.p_post]  # (T_post, X, Y, Z), uint8

        # =========================
        #   Option A: time as channels
        # =========================
        # input_hist: (T_pre, X, Y, Z) -> build (N, T_pre) and coords (N, 3)
        # We ensure unique (z,y,x). Each feature vector is [frame0, frame1, ..., frame_{p_pre-1}]
        # where frame_{p_pre-1} is the *current* time t=0.
        input_hist = (input_hist > 0)  # boolean
        flat_TV = input_hist.reshape(self.p_pre, -1)           # (T_pre, X*Y*Z)
        site_mask = flat_TV.any(axis=0)                        # (X*Y*Z,)
        ids = np.flatnonzero(site_mask)                        # [N_sites]
        if ids.size == 0:
            # Corner case: no occupied voxels in the entire history; create an empty sparse sample.
            voxel_features = np.zeros((0, self.p_pre), dtype=np.float32)
            voxel_coords = np.zeros((0, 3), dtype=np.int32)
        else:
            feats = flat_TV[:, site_mask].T.astype(np.float32)  # (N, T_pre)
            voxel_features = feats                               # (N, p_pre)
            voxel_coords = self._flat_ids_to_zyx(ids, X, Y, Z)   # (N, 3) (z, y, x)

        # --- Convert dense tensors ---
        label_tensor = torch.tensor(label)            # (T_post, X, Y, Z), dtype inferred (usually int64/uint8)
        invalid_tensor = torch.tensor(invalid).bool() # (T_post, X, Y, Z) -> boolean mask


        return {
            'voxel_features': voxel_features,   # np.float32 [N, p_pre]
            'voxel_coords': voxel_coords,       # np.int32  [N, 3] (z, y, x)
            'labels': label_tensor,             # torch.Tensor [T_post, X, Y, Z]
            'invalid': invalid_tensor,          # torch.BoolTensor [T_post, X, Y, Z]
        }

    def get_voxel_size(self):
        # Return (X, Y, Z, T_post) to match your original helper
        return self.X, self.Y, self.Z, self.p_post

    def get_class_fc(self):
        if self.semantic:
            return self.config['label_fc']
        else:
            fc_dict = dict(self.config['label_fc'])  # avoid mutating config
            for i in range(2, len(fc_dict)):
                fc_dict[1] += fc_dict[i]
                del fc_dict[i]
            return fc_dict

    # ---------------------------
    # Collate fn for DataLoader
    # ---------------------------
    @staticmethod
    def collate_batch(batch_list):
        """
        Produces a dictionary ready for the spconv model:
            - voxel_features: concatenated across batch  -> np.float32 [N_all, C]
            - voxel_coords:   padded with batch index    -> np.int32   [N_all, 4]  (b, z, y, x)
            - labels/invalid: stacked dense tensors      -> torch.Tensor [B, T_post, X, Y, Z]
            - batch_size:     int
        """
        from collections import defaultdict
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for k, v in cur_sample.items():
                data_dict[k].append(v)

        batch_size = len(batch_list)
        ret = {}

        # Sparse features & coords
        if 'voxel_features' in data_dict:
            ret['voxel_features'] = np.concatenate(data_dict['voxel_features'], axis=0) \
                if len(data_dict['voxel_features']) > 1 else data_dict['voxel_features'][0]

        if 'voxel_coords' in data_dict:
            coords_with_batch = []
            for b_idx, coor in enumerate(data_dict['voxel_coords']):
                # pad a leading batch column
                if coor.size == 0:
                    continue
                coor = np.asarray(coor, dtype=np.int32)
                coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=b_idx)
                coords_with_batch.append(coor_pad)
            if len(coords_with_batch) == 0:
                # handle fully-empty batch gracefully
                ret['voxel_coords'] = np.zeros((0, 4), dtype=np.int32)
            else:
                ret['voxel_coords'] = np.concatenate(coords_with_batch, axis=0)

        # Dense labels/invalid
        for key in ['labels', 'invalid']:
            if key in data_dict:
                stacked = torch.stack(data_dict[key], dim=0)  # (B, T_post, X, Y, Z)
                ret[key] = stacked

        ret['batch_size'] = batch_size
        return ret
