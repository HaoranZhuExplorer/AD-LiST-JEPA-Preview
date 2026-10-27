import os
import yaml
from glob import glob
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LyftDataset(Dataset):
    """
    Occ4D-Lyft dataset (time-separated sparse inputs + dense labels/invalid).
    Compatible with per-frame sparse 3D encoders (spconv/OpenPCDet style).
    """

    def __init__(self, root_dir: str, split: str = 'train', p_pre: int = 10, p_post: int = 10, semantic: bool = False):
        """
        Args:
            root_dir: dataset root (contains 'voxel/' subdir)
            split: 'train' | 'valid' | 'test'
            p_pre: number of frames BEFORE current; dataset includes current -> internally p_pre+1
            p_post: number of frames AFTER current; dataset includes current -> internally p_post+1
            semantic: keep semantic class fc (not used for binary OCF)
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

        # include current frame for both sides
        self.p_pre = p_pre + 1      # history + current
        self.p_post = p_post + 1    # current + future
        self.semantic = semantic

        with open("configs/lyft.yaml", 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self._load_data()

    def _load_data(self):
        self.data_path: List[str] = []
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
                label_path = os.path.join(seq_dir, f"{sample_id}_label.npy")
                invalid_path = os.path.join(seq_dir, f"{sample_id}_invalid.npy")
                if os.path.isfile(label_path) and os.path.isfile(invalid_path):
                    self.data_path.append(os.path.join(os.path.abspath(seq_dir), sample_id))

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            {
              'coords_per_t': [list length p_pre] each [N_t, 3] int32 (z,y,x),
              't_secs':        [p_pre] float32 times (…,-0.2,-0.1,0.0),
              'labels':        [T_post, X, Y, Z] int8/uint8,
              'invalid':       [T_post, X, Y, Z] int8/uint8,
              # optional sparse for vis/debug:
              'label_voxels':        [N_lbl, 4] float32 (z,y,x,t_sec),
              'label_voxel_coords':  [N_lbl, 3] int32,
              'invalid_voxels':      [N_inv, 4] float32,
              'invalid_voxel_coords':[N_inv, 3] int32,
              # meta
              'grid_size_zyx': [3] int32,
              'num_input_frames': int,
              'num_future_frames': int,
            }
        """
        stub = self.data_path[idx]

        # ----- LABELS (dense, not bit-packed) -----
        data_label = np.load(stub + "_label.npy")           # [T_post_total, X, Y, Z]
        label = data_label[:self.p_post]                     # [T_post, X, Y, Z]
        T_post_total, X, Y, Z = data_label.shape

        # ----- INPUT (bit-packed) -----
        data_input_arr = np.load(stub + "_input.npy")       # packed uint8
        data_input_bits = np.unpackbits(data_input_arr)
        T_in_total = int(data_input_bits.size // (X * Y * Z))
        if T_in_total < self.p_pre:
            raise ValueError(f"Not enough input frames: have {T_in_total}, need {self.p_pre}")
        data_input = data_input_bits.reshape(T_in_total, X, Y, Z).astype(np.uint8)
        input_hist = data_input[-self.p_pre:]               # [T_pre, X, Y, Z] (history + current)

        # ----- INVALID (bit-packed) -----
        data_invalid_arr = np.load(stub + "_invalid.npy")
        data_invalid_bits = np.unpackbits(data_invalid_arr)
        T_inv_total = int(data_invalid_bits.size // (X * Y * Z))
        if T_inv_total < self.p_post:
            raise ValueError(f"Not enough invalid frames: have {T_inv_total}, need {self.p_post}")
        invalid = data_invalid_bits.reshape(T_inv_total, X, Y, Z)[:self.p_post].astype(np.uint8)
        #print("debug dense input", np.sum(input_hist), np.sum(label), np.sum(invalid))

        # ----- Sparse INPUT per timestamp -----
        coords_per_t: List[np.ndarray] = []   # list of [N_t, 3] int32 (z,y,x)
        t_secs: List[float] = []

        # Convention: last input frame is current t=0s; step = 0.1s (OCFBench)
        for t in range(self.p_pre):
            occ_xyz = np.argwhere(input_hist[t] > 0)  # [N_t, 3] (x,y,z)
           
            coords_t = occ_xyz[:, [2, 1, 0]].astype(np.int32)  # -> (z,y,x)
            coords_per_t.append(coords_t)
            t_secs.append(0.1 * (t - (self.p_pre - 1)))  # ..., -0.2, -0.1, 0.0

        # ----- Optional sparse LABEL/INVALID (for visualization) -----
        label_occ_idx = np.argwhere(label > 0)  # (t, x, y, z)
        
        label_voxels = label_occ_idx[:, [3, 2, 1, 0]].astype(np.float32)  # (z,y,x,tIdx)
        label_voxels[:, 3] = 0.1 * label_occ_idx[:, 0].astype(np.float32)
        label_voxel_coords = label_voxels[:, :3].astype(np.int32)

        invalid_occ_idx = np.argwhere(invalid > 0)  # (t, x, y, z)
        
        invalid_voxels = invalid_occ_idx[:, [3, 2, 1, 0]].astype(np.float32)
        invalid_voxels[:, 3] = 0.1 * invalid_occ_idx[:, 0].astype(np.float32)
        invalid_voxel_coords = invalid_voxels[:, :3].astype(np.int32)

        # ----- Dense tensors for loss -----
        label_tensor = torch.from_numpy(label)         # [T_post, X, Y, Z]
        invalid_tensor = torch.from_numpy(invalid)     # [T_post, X, Y, Z]
        input_dense_tensor = torch.from_numpy(input_hist).to(torch.float32)  # [T_pre, X, Y, Z]

        #print("debug dataloader label", label_tensor.shape, torch.sum(label_tensor), input_dense_tensor.shape, torch.sum(input_dense_tensor))
        #print("debug dataloader input", input_hist.shape, np.sum(input_hist))
        #print("debug dataloader label_voxels", label_voxels.shape)

        return {
            'coords_per_t': coords_per_t,                        # list of [N_t, 3] (z,y,x)
            't_secs': np.asarray(t_secs, dtype=np.float32),      # [p_pre]
 
            'input_dense': input_dense_tensor,                            # [T_pre, X, Y, Z]
            'labels': label_tensor,                               # [T_post, X, Y, Z]
            'invalid': invalid_tensor,                            # [T_post, X, Y, Z]

            'label_voxels': label_voxels,                        # [N_lbl, 4]
            'label_voxel_coords': label_voxel_coords,            # [N_lbl, 3]
            'invalid_voxels': invalid_voxels,                    # [N_inv, 4]
            'invalid_voxel_coords': invalid_voxel_coords,        # [N_inv, 3]

            'grid_size_zyx': np.array([Z, Y, X], dtype=np.int32),
            'num_input_frames': self.p_pre,
            'num_future_frames': self.p_post,
        }

    def get_voxel_size(self) -> Tuple[int, int, int, int]:
        # For API compatibility; actual XYZ are inferred per sample from labels.
        return 512, 256, 32, self.p_post

    def get_class_fc(self) -> Dict[int, int]:
        if self.semantic:
            return self.config['label_fc']
        else:
            fc_dict = dict(self.config['label_fc'])  # shallow copy
            for i in range(2, len(fc_dict)):
                fc_dict[1] += fc_dict[i]
                del fc_dict[i]
            return fc_dict

    # ---------------- Collate ---------------- #

    @staticmethod
    def collate_batch(
        batch_list: List[Dict[str, Any]],
        feat_mode: str = 'ones',         # 'idx' | 'idx_norm' | 'ones'
        append_time: bool = False,      # if True, append t_sec as extra feature
    ) -> Dict[str, Any]:
        """
        Packs time as part of the batch index for per-frame sparse 3D convs.

        Output:
          - input_coords_bt: [N_all, 4] int32 (bt, z, y, x), bt = b * p_pre + t
          - input_feats:     [N_all, C] float32; C=3 ('idx'/'idx_norm') or 1 ('ones') [+1 if append_time]
          - bt_to_b_t:       [B * p_pre, 2] int32 mapping bt -> (b, t)
          - t_secs:          [B, p_pre] float32
          - labels:          [B, T_post, X, Y, Z]
          - invalid:         [B, T_post, X, Y, Z]
          - spatial_shape_zyx: [3] int32 (Z, Y, X)
          - batch_size:      int B
          - batch_size_time: int B * p_pre
        """
        B = len(batch_list)
        assert B > 0
        p_pre = batch_list[0]['num_input_frames']
        grid_zyx = batch_list[0]['grid_size_zyx']  # np.array([Z,Y,X])
        Z, Y, X = int(grid_zyx[0]), int(grid_zyx[1]), int(grid_zyx[2])

        all_coords, all_feats, bt_to_b_t, t_grid = [], [], [], []

        def _coords_to_feats(zyx: np.ndarray, t_sec: float) -> np.ndarray:
            """Build features from coords as requested: same as coordinate."""
            if feat_mode == 'ones':
                feats = np.ones((zyx.shape[0], 1), dtype=np.float32)
            elif feat_mode == 'idx':
                # raw voxel indices in (x,y,z)
                feats = zyx[:, [2, 1, 0]].astype(np.float32)
            elif feat_mode == 'idx_norm':
                # center bin (+0.5), normalize to [-1, 1]
                x = ((zyx[:, 2].astype(np.float32) + 0.5) / X) * 2.0 - 1.0
                y = ((zyx[:, 1].astype(np.float32) + 0.5) / Y) * 2.0 - 1.0
                z = ((zyx[:, 0].astype(np.float32) + 0.5) / Z) * 2.0 - 1.0
                feats = np.stack([x, y, z], axis=1)
            else:
                raise ValueError(f"Unknown feat_mode={feat_mode}")

            if append_time:
                tcol = np.full((feats.shape[0], 1), float(t_sec), dtype=np.float32)
                feats = np.concatenate([feats, tcol], axis=1)
            return feats

        for b, sample in enumerate(batch_list):
            assert sample['num_input_frames'] == p_pre, "All samples in batch must share p_pre"
            coords_list = sample['coords_per_t']           # list of [N_t,3] (z,y,x)
            t_secs = sample['t_secs']                      # [p_pre]
            t_grid.append(t_secs)

            for t, coors_t in enumerate(coords_list):
                bt = b * p_pre + t
                bt_to_b_t.append([b, t])
                if coors_t.shape[0] == 0:
                    continue
                # pad with bt in first column -> (bt, z, y, x)
                coor_bt = np.pad(coors_t, ((0, 0), (1, 0)), mode='constant', constant_values=bt)
                all_coords.append(coor_bt.astype(np.int32))
                all_feats.append(_coords_to_feats(coors_t, t_secs[t]))

       
        input_coords_bt = torch.from_numpy(np.concatenate(all_coords, axis=0)).to(torch.int32)
        input_feats = torch.from_numpy(np.concatenate(all_feats, axis=0)).to(torch.float32)

        labels = torch.stack([s['labels'] for s in batch_list], dim=0)    # [B, T_post, X, Y, Z]
        invalid = torch.stack([s['invalid'] for s in batch_list], dim=0)  # [B, T_post, X, Y, Z]
        input_dense = torch.stack([s['input_dense'] for s in batch_list], dim=0)

        #print("debug", input_feats[:10,:], input_coords_bt[:10,:])

        ret = {
            'input_coords_bt': input_coords_bt,                         # [N, 4] (bt,z,y,x)
            'input_feats': input_feats,                                 # [N, C]
            'bt_to_b_t': torch.tensor(bt_to_b_t, dtype=torch.int32),    # [B*p_pre, 2]
            't_secs': torch.from_numpy(np.stack(t_grid, axis=0)),       # [B, p_pre]
            'labels': labels,
            'invalid': invalid,
            'input_dense': input_dense,
            'spatial_shape_zyx': torch.from_numpy(grid_zyx),            # [3] (Z,Y,X)
            'batch_size': B,
            'batch_size_time': B * p_pre,
        }
        return ret
    

if __name__ == "__main__":
    import argparse
    import numpy as np
    import os

    # Headless-safe plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _parse_stub_parts(stub_path: str):
        """
        Given a stub like ".../voxel/0000/0069", return ("0000", "0069").
        """
        seq_id = os.path.basename(os.path.dirname(stub_path))
        sample_id = os.path.basename(stub_path)
        return seq_id, sample_id

    def _save_empty_notice(out_dir: str, prefix: str):
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No occupied voxels", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(os.path.join(out_dir, f"{prefix}_empty.png"),
                    bbox_inches="tight", dpi=200)
        plt.close(fig)

    def _coords_to_bev_mask_xy(coords_zyx: np.ndarray, X: int, Y: int) -> np.ndarray:
        """
        Build a BEV (Y, X) binary mask from sparse voxel coords in (z, y, x).
        coords_zyx: [N, 3] int32
        """
        bev = np.zeros((Y, X), dtype=np.uint8)
        if coords_zyx.size == 0:
            return bev
        xy = coords_zyx[:, [2, 1]].astype(np.int32)  # (x,y)
        xy[:, 0] = np.clip(xy[:, 0], 0, X - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, Y - 1)
        bev[xy[:, 1], xy[:, 0]] = 1
        return bev

    def _voxels_time_to_bev_mask_xy(voxels_zyxt: np.ndarray, X: int, Y: int, t_val: float, round_decimals: int) -> np.ndarray:
        """
        From a [N,4] array (z,y,x,t_sec), pick time==t_val (rounded), and build BEV mask (Y,X).
        """
        if voxels_zyxt.size == 0:
            return np.zeros((Y, X), dtype=np.uint8)

        t_all = np.round(voxels_zyxt[:, 3].astype(np.float64), round_decimals)
        mask = (t_all == np.round(float(t_val), round_decimals))
        if not np.any(mask):
            return np.zeros((Y, X), dtype=np.uint8)

        xy = voxels_zyxt[mask][:, [2, 1]].astype(np.int32)  # (x,y)
        xy[:, 0] = np.clip(xy[:, 0], 0, X - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, Y - 1)
        bev = np.zeros((Y, X), dtype=np.uint8)
        bev[xy[:, 1], xy[:, 0]] = 1
        return bev

    def _plot_bev(bev_mask: np.ndarray, out_path: str, title: str):
        """
        Render a BEV occupancy mask (Y,X) to PNG.
        """
        Y, X = bev_mask.shape
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(bev_mask, origin="lower", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("X (voxel index)")
        ax.set_ylabel("Y (voxel index)")
        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.grid(False)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    parser = argparse.ArgumentParser(
        description="Visualize BEV occupancy per time step for a single LyftDataset sample."
    )
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Dataset root (contains 'voxel/' etc.)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--idx", type=int, required=True,
                        help="Index of the sample within the split to visualize.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save generated figures.")
    parser.add_argument("--p_pre", type=int, default=10,
                        help="Number of frames before current (dataset includes current internally).")
    parser.add_argument("--p_post", type=int, default=10,
                        help="Number of frames after current (dataset includes current internally).")
    parser.add_argument("--semantic", action="store_true",
                        help="Use semantic setting (not needed for plotting).")
    parser.add_argument("--round_decimals", type=int, default=3,
                        help="Decimals for rounding time values when grouping (default: 3).")
    args = parser.parse_args()

    # Build dataset
    ds = LyftDataset(
        root_dir=args.root_dir,
        split=args.split,
        p_pre=args.p_pre,
        p_post=args.p_post,
        semantic=args.semantic
    )

    if len(ds) == 0:
        raise RuntimeError("Dataset is empty for the given configuration.")

    if not (0 <= args.idx < len(ds)):
        raise IndexError(f"idx={args.idx} out of range for split '{args.split}' with length {len(ds)}.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load sample
    sample = ds[args.idx]

    # Pull grid size from the returned dict (Z,Y,X) and build name prefix
    Z, Y, X = int(sample["grid_size_zyx"][0]), int(sample["grid_size_zyx"][1]), int(sample["grid_size_zyx"][2])
    seq_id, sample_id = _parse_stub_parts(ds.data_path[args.idx])
    prefix = f"seq{seq_id}_sample{sample_id}"

    # ----------------------------
    # 1) INPUT (history..0s) from coords_per_t
    # ----------------------------
    coords_per_t = sample["coords_per_t"]               # list of [N_t,3] (z,y,x)
    t_secs_input = sample["t_secs"].astype(np.float64)  # [p_pre], ..., -0.2, -0.1, 0.0

    if len(coords_per_t) == 0:
        _save_empty_notice(args.out_dir, prefix + "_input")
        print(f"[INFO] No input frames for {prefix}. Saved empty placeholder.")
    else:
        print(f"[INFO] Visualizing {prefix} (INPUT): {len(coords_per_t)} time slices "
              f"from t={t_secs_input.min():+.1f}s to t={t_secs_input.max():+.1f}s")

        for coords_zyx, t_val in zip(coords_per_t, t_secs_input):
            bev = _coords_to_bev_mask_xy(coords_zyx, X=X, Y=Y)

            # Force "-0.0" for the current frame (for naming consistency)
            t_round = np.round(t_val, args.round_decimals)
            if np.isclose(t_round, 0.0):
                t_str = "-0.0"
            else:
                t_str = f"{t_round:+.{args.round_decimals}f}"

            out_path = os.path.join(args.out_dir, f"{prefix}_input_t{t_str}s.png")
            _plot_bev(bev, out_path, title=f"BEV Occupancy — {prefix} @ t={t_round:+.1f}s (INPUT)")

    # ----------------------------
    # 2) LABELS (0..future) from label_voxels
    # ----------------------------
    label_voxels = sample["label_voxels"]  # [N,4] (z,y,x,t_sec)
    if label_voxels.size == 0:
        _save_empty_notice(args.out_dir, prefix + "_labels")
        print(f"[INFO] No occupied voxels for {prefix} (LABELS). Saved empty placeholder.")
    else:
        t_all_label = np.round(label_voxels[:, 3].astype(np.float64), args.round_decimals)
        unique_times_label = np.unique(t_all_label)
        print(f"[INFO] Visualizing {prefix} (LABELS): {len(unique_times_label)} time slices "
              f"from t={unique_times_label.min():+.1f}s to t={unique_times_label.max():+.1f}s")

        for t_val in unique_times_label:
            bev = _voxels_time_to_bev_mask_xy(
                voxels_zyxt=label_voxels,
                X=X, Y=Y,
                t_val=float(t_val),
                round_decimals=args.round_decimals
            )
            t_str = f"{t_val:+.{args.round_decimals}f}"
            out_path = os.path.join(args.out_dir, f"{prefix}_labels_t{t_str}s.png")
            _plot_bev(bev, out_path, title=f"BEV Occupancy — {prefix} @ t={t_val:+.1f}s (LABELS)")

    # ----------------------------
    # 3) INVALID (0..future) from invalid_voxels
    # ----------------------------
    invalid_voxels = sample["invalid_voxels"]  # [N,4] (z,y,x,t_sec)
    if invalid_voxels.size == 0:
        _save_empty_notice(args.out_dir, prefix + "_invalid")
        print(f"[INFO] No occupied voxels for {prefix} (INVALID). Saved empty placeholder.")
    else:
        t_all_invalid = np.round(invalid_voxels[:, 3].astype(np.float64), args.round_decimals)
        unique_times_invalid = np.unique(t_all_invalid)
        print(f"[INFO] Visualizing {prefix} (INVALID): {len(unique_times_invalid)} time slices "
              f"from t={unique_times_invalid.min():+.1f}s to t={unique_times_invalid.max():+.1f}s")

        for t_val in unique_times_invalid:
            bev = _voxels_time_to_bev_mask_xy(
                voxels_zyxt=invalid_voxels,
                X=X, Y=Y,
                t_val=float(t_val),
                round_decimals=args.round_decimals
            )
            t_str = f"{t_val:+.{args.round_decimals}f}"
            out_path = os.path.join(args.out_dir, f"{prefix}_invalid_t{t_str}s.png")
            _plot_bev(bev, out_path, title=f"BEV Occupancy — {prefix} @ t={t_val:+.1f}s (INVALID)")

    print(f"[DONE] Saved images to: {args.out_dir}")
