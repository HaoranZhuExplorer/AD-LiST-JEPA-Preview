# v0
import os
import yaml
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class LyftDataset(Dataset):
    """Occ4D-Lyft dataset class."""
    def __init__(self, root_dir, split='train', p_pre=10, p_post=10, semantic=False):
        """
        Args:
            root_dir (string): Directory with all the data.
            split (string): 'train' or 'test' split.
            p_pre (int): Number of frames before the current frame.
            p_post (int): Number of frames after the current frame.
        """
        self.root_dir = root_dir
        if split == 'train':
            self.split = list(range(120))
        elif split == 'valid':
            self.split = list(range(120, 150))
        elif split == 'test':
            self.split = list(range(150, 180))
        else:
            raise ValueError('Invalid split name: {}'.format(split))
        
        self.p_pre = p_pre + 1  # +1 for the current frame
        self.p_post = p_post + 1
        self.semantic = semantic
        
        with open("configs/lyft.yaml", 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self._load_data()

    def _load_data(self):
        self.data_path = []

        voxel_root = os.path.join(self.root_dir, "voxel")
    
        for seq_id in self.split:
            seq_str = f"{seq_id:04d}"
            seq_dir = os.path.join(voxel_root, seq_str)
            if not os.path.isdir(seq_dir):
                continue

            # find all *_input.npy in this sequence
            for input_path in sorted(glob(os.path.join(seq_dir, "*_input.npy"))):
                name = os.path.basename(input_path)
                if not name.endswith("_input.npy"):
                    continue
                sample_id = name[:-10]  # strip "_input.npy", e.g. "0069"

                label_path   = os.path.join(seq_dir, f"{sample_id}_label.npy")
                invalid_path = os.path.join(seq_dir, f"{sample_id}_invalid.npy")

                # only keep if all three exist
                if os.path.isfile(label_path) and os.path.isfile(invalid_path):
                    # absolute stub: "/Occ4D-Lyft/voxel/0000/0069"
                    stub = os.path.join(os.path.abspath(seq_dir), sample_id)
                    self.data_path.append(stub)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        # --- Load LABELS (dense) ---
        data_label = np.load(self.data_path[idx] + "_label.npy")
        label = data_label[:self.p_post]  # [T_post, X, Y, Z] (assumed non bit-packed)

        # --- Load INPUT (bit-packed) and slice the last p_pre frames (history+current) ---
        data_input_arr = np.load(self.data_path[idx] + "_input.npy")
        data_input_bits = np.unpackbits(data_input_arr)
        data_input = data_input_bits.reshape(11, 512, 256, 32)  # TODO: shape hard coding for now
        input = data_input[-self.p_pre:, :, :, :]                # [T_pre, X, Y, Z]

        # --- Load INVALID (bit-packed) and take first p_post frames (current+future) ---
        data_invalid_arr = np.load(self.data_path[idx] + "_invalid.npy")
        data_invalid_bits = np.unpackbits(data_invalid_arr)
        data_invalid = data_invalid_bits.reshape(11, 512, 256, 32)  # TODO: shape hard coding for now
        invalid = data_invalid[:self.p_post, :, :, :]               # [T_post, X, Y, Z]

        # --- Sparse conversion: INPUT -> voxels / voxel_coords ---
        # find occupied locations; argwhere returns (t, x, y, z)
        occ_idx = np.argwhere(input > 0)
        
        # permute to (z, y, x, t) in one step
        voxels = occ_idx[:, [3, 2, 1, 0]].astype(np.float32)
        # time encoding for INPUT: negative history up to 0.0s, 0.1s per frame
        # input frames are indexed so that the last frame is current (index self.p_pre-1),
        # which corresponds to t=0.0s after the shift by (t - (self.p_pre-1)).
        voxels[:, 3] = 0.1 * (voxels[:, 3] - (self.p_pre - 1))
        voxel_coords = voxels[:, 0:3].astype(np.int32)  # [N, 3]

        # --- Sparse conversion: LABEL -> label_voxels / label_voxel_coords ---
        # We treat any label > 0 as occupied for visualization; time is 0.1s * t (0.0, +0.1, ...)
        label_occ_idx = np.argwhere(label > 0)
        if label_occ_idx.size == 0:
            label_voxels = np.zeros((0, 4), dtype=np.int32)
            label_voxel_coords = np.zeros((0, 3), dtype=np.int32)
        else:
            label_voxels = label_occ_idx[:, [1, 2, 3, 0]].astype(np.float32)
            label_voxels[:, 3] = 0.1 * label_occ_idx[:, 0].astype(np.float32)
            label_voxel_coords = label_voxels[:, 0:3].astype(np.int32)

        # --- Sparse conversion: INVALID -> invalid_voxels / invalid_voxel_coords ---
        invalid_occ_idx = np.argwhere(invalid > 0)
        if invalid_occ_idx.size == 0:
            invalid_voxels = np.zeros((0, 4), dtype=np.int32)
            invalid_voxel_coords = np.zeros((0, 3), dtype=np.int32)
        else:
            invalid_voxels = invalid_occ_idx[:, [1, 2, 3, 0]].astype(np.float32)
            invalid_voxels[:, 3] = 0.1 * invalid_occ_idx[:, 0].astype(np.float32)
            invalid_voxel_coords = invalid_voxels[:, 0:3].astype(np.int32)

        # Dense tensors (unchanged)
        label_tensor = torch.tensor(label)
        invalid_tensor = torch.from_numpy(invalid)

        return {
            'voxel_features': voxels, # [N, 4], (z, y, x, t)
            'voxel_coords': voxel_coords, # [N, 3], (z, y, x)
            'label_voxels': label_voxels, 
            'label_voxel_coords': label_voxel_coords,
            'invalid_voxels': invalid_voxels,
            'invalid_voxel_coords': invalid_voxel_coords,
            'labels': label_tensor,
            'invalid': invalid_tensor,
        }

    def get_voxel_size(self):
        X, Y, Z = 512, 256, 32  # TODO: shape hard coding for now
        T = self.p_post
        return X, Y, Z, T

    def get_class_fc(self):
        if self.semantic:
            return self.config['label_fc']
        else:
            fc_dict = self.config['label_fc']
            for i in range(2, len(fc_dict)):
                fc_dict[1] += fc_dict[i]
                del fc_dict[i]
            return fc_dict
        
    
    @staticmethod
    def collate_batch(batch_list):
        from collections import defaultdict
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            if key =='voxel_features':
                ret[key] = np.concatenate(val, axis=0)
            elif key == 'voxel_coords':
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)

            elif key in ['labels', 'invalid']:
                list_ = []
                for val_i in val:
                    list_.append(val_i)
                ret[key] = torch.from_numpy(np.stack(list_, axis=0))

        ret['batch_size'] = batch_size

        return ret
        

# use as for visualization:
# python data/lyft_sparse.py --root_dir /Occ4D-Lyft --split train --idx 100 --out_dir <your_path>/Occ4cast_mine/baselines/debug
if __name__ == "__main__":
    import argparse
    import os
    import numpy as np

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

    def _plot_bev_for_time(voxels_t: np.ndarray, grid_size_xy, out_path: str, title: str):
        """
        Render a BEV occupancy mask for the given set of voxels at a single time.
        voxels_t: [N, 4] with columns (x, y, z, t)
        grid_size_xy: (X, Y)
        out_path: file path to save PNG
        """
        X, Y = grid_size_xy
        if voxels_t.size == 0:
            # still save an empty map for this time step
            bev = np.zeros((Y, X), dtype=np.uint8)
        else:
            # Collapse Z: unique (x, y) pairs are enough for BEV
            xy = voxels_t[:, :2].astype(np.int32)
            # Clamp indices to valid range just in case
            xy[:, 0] = np.clip(xy[:, 0], 0, X - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, Y - 1)

            bev = np.zeros((Y, X), dtype=np.uint8)  # rows=Y, cols=X
            bev[xy[:, 1], xy[:, 0]] = 1

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(bev, origin="lower", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("X (voxel index)")
        ax.set_ylabel("Y (voxel index)")
        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.grid(False)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    parser = argparse.ArgumentParser(
        description="Test: visualize BEV occupancy per time step for a single sample."
    )
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Occ4D-Lyft dataset root (contains 'voxel/' etc.)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"],
                        help="Dataset split.")
    parser.add_argument("--idx", type=int, required=True,
                        help="Index of the sample within the split to visualize.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save the generated figures.")
    parser.add_argument("--p_pre", type=int, default=10,
                        help="Number of frames before current (dataset expects 10 by default).")
    parser.add_argument("--p_post", type=int, default=10,
                        help="Number of frames after current (for labels/invalid).")
    parser.add_argument("--semantic", action="store_true",
                        help="Use semantic setting for class fc (not needed for plotting).")
    parser.add_argument("--round_decimals", type=int, default=3,
                        help="Decimals for rounding time values to group frames (default: 3).")
    args = parser.parse_args()

    # Build dataset and guardrails
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
    voxels = sample["voxel_features"][:, [2, 1, 0, 3]]                # [N, 4], (x, y, z, t_seconds) for INPUT (history..0s)
    label_voxels = sample["label_voxels"]      # [N, 4], for LABELS (0s..future)
    invalid_voxels = sample["invalid_voxels"]  # [N, 4], for INVALID (0s..future)
    print("debug", voxels.shape, label_voxels.shape, invalid_voxels.shape)

    X, Y, _, _ = ds.get_voxel_size()
    seq_id, sample_id = _parse_stub_parts(ds.data_path[args.idx])
    prefix = f"seq{seq_id}_sample{sample_id}"

    # ----------------------------
    # 1) INPUT (history) BEV plots
    # ----------------------------
    if voxels.size == 0:
        _save_empty_notice(args.out_dir, prefix)
        print(f"[INFO] No occupied voxels for {prefix} (INPUT). Saved empty placeholder.")
        n_input_imgs = 1
    else:
        # Group by time (the last column). __getitem__ encodes t in seconds with 0.1s steps.
        # We round to avoid floating-point artifacts (e.g., -0.30000000004).
        t_all = np.round(voxels[:, 3].astype(np.float64), args.round_decimals)
        unique_times = np.unique(t_all)
        # Keep the original array, but use the rounded times for masking
        voxels_with_t_rounded = np.c_[voxels[:, :3], t_all]

        print(f"[INFO] Visualizing {prefix} (INPUT): {len(unique_times)} time slices "
              f"from t={unique_times.min():.1f}s to t={unique_times.max():.1f}s")

        for t in unique_times:
            mask = voxels_with_t_rounded[:, 3] == t
            voxels_t = voxels[mask]

            # File naming: keep the original negative-zero convention for INPUT
            if np.isclose(t, 0.0):
                t_str = "-0.0"  # force negative zero for the current frame in INPUT
            else:
                t_str = f"{t:+.1f}"

            out_path = os.path.join(args.out_dir, f"{prefix}_t{t_str}s.png")
            title = f"BEV Occupancy — {prefix} @ t={t:+.1f}s (INPUT)"
            _plot_bev_for_time(
                voxels_t=voxels_t,
                grid_size_xy=(X, Y),
                out_path=out_path,
                title=title
            )
        n_input_imgs = len(unique_times)

    # ----------------------------
    # 2) LABELS (current+future) BEV plots
    # ----------------------------
    if label_voxels.size == 0:
        _save_empty_notice(args.out_dir, prefix + "_labels")
        print(f"[INFO] No occupied voxels for {prefix} (LABELS). Saved empty placeholder.")
        n_label_imgs = 1
    else:
        t_all_label = np.round(label_voxels[:, 3].astype(np.float64), args.round_decimals)
        unique_times_label = np.unique(t_all_label)
        label_with_t_rounded = np.c_[label_voxels[:, :3], t_all_label]

        print(f"[INFO] Visualizing {prefix} (LABELS): {len(unique_times_label)} time slices "
              f"from t={unique_times_label.min():+.1f}s to t={unique_times_label.max():+.1f}s")

        for t in unique_times_label:
            mask = label_with_t_rounded[:, 3] == t
            voxels_t = label_voxels[mask]

            t_str = f"{t:+.1f}"  # positive or zero times are fine as-is
            out_path = os.path.join(args.out_dir, f"{prefix}_labels_t{t_str}s.png")
            title = f"BEV Occupancy — {prefix} @ t={t:+.1f}s (LABELS)"
            _plot_bev_for_time(
                voxels_t=voxels_t,
                grid_size_xy=(X, Y),
                out_path=out_path,
                title=title
            )
        n_label_imgs = len(unique_times_label)

    # ----------------------------
    # 3) INVALID (current+future) BEV plots
    # ----------------------------
    if invalid_voxels.size == 0:
        _save_empty_notice(args.out_dir, prefix + "_invalid")
        print(f"[INFO] No occupied voxels for {prefix} (INVALID). Saved empty placeholder.")
        n_invalid_imgs = 1
    else:
        t_all_invalid = np.round(invalid_voxels[:, 3].astype(np.float64), args.round_decimals)
        unique_times_invalid = np.unique(t_all_invalid)
        invalid_with_t_rounded = np.c_[invalid_voxels[:, :3], t_all_invalid]

        print(f"[INFO] Visualizing {prefix} (INVALID): {len(unique_times_invalid)} time slices "
              f"from t={unique_times_invalid.min():+.1f}s to t={unique_times_invalid.max():+.1f}s")

        for t in unique_times_invalid:
            mask = invalid_with_t_rounded[:, 3] == t
            voxels_t = invalid_voxels[mask]

            t_str = f"{t:+.1f}"
            out_path = os.path.join(args.out_dir, f"{prefix}_invalid_t{t_str}s.png")
            title = f"BEV Occupancy — {prefix} @ t={t:+.1f}s (INVALID)"
            _plot_bev_for_time(
                voxels_t=voxels_t,
                grid_size_xy=(X, Y),
                out_path=out_path,
                title=title
            )
        n_invalid_imgs = len(unique_times_invalid)

    print(f"[DONE] Saved images to: {args.out_dir} "
          f"(INPUT: {n_input_imgs}, LABELS: {n_label_imgs}, INVALID: {n_invalid_imgs})")
