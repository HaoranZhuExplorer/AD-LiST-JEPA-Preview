# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Modified for Occupancy Completion / Object Densification with transform.py output format
#
# Output matches transform.py contract:
#   - points:  dict of (4, N) float32  [x;y;z;1]
#   - labels:  dict of (N,) uint8      default=1 (UNLABELED)
#   - poses:   dict of (4, 4) float32  vehicle2global
#
# IMPORTANT: Cyclist label id choice:
#   - transform.py defines BICYCLE=6 and BICYCLIST=7.
#   - Waymo "Cyclist" is typically a rider => default here uses 7.
#   - If you want 6 instead, change LABEL_REMAP['Cyclist'] to 6.

import os
import pickle
import copy
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm

from ..dataset import DatasetTemplate
from ...utils import common_utils

# -----------------------------
# transform.py-style label ids
# -----------------------------
UNLABELED_ID = 1
LABEL_REMAP = {
    "Vehicle": 2,      # VEHICLE
    "Pedestrian": 3,   # PEDESTRIAN
    "Cyclist": 7,      # BICYCLIST (set to 6 if you want BICYCLE)
}


def get_box_corners_np(boxes7: np.ndarray) -> np.ndarray:
    """
    Args:
        boxes7: (M, 7) [x, y, z, dx, dy, dz, heading] (center-based)
    Returns:
        corners: (M, 8, 3) corners in the SAME frame as boxes
                 with consistent ordering across frames (needed for SVD correspondence).
    """
    assert boxes7.ndim == 2 and boxes7.shape[1] == 7
    M = boxes7.shape[0]
    corners = np.zeros((M, 8, 3), dtype=np.float32)

    # canonical corners in box local frame
    dx = boxes7[:, 3].astype(np.float32)
    dy = boxes7[:, 4].astype(np.float32)
    dz = boxes7[:, 5].astype(np.float32)

    x_c = np.stack([ dx/2,  dx/2, -dx/2, -dx/2,  dx/2,  dx/2, -dx/2, -dx/2], axis=1)
    y_c = np.stack([ dy/2, -dy/2, -dy/2,  dy/2,  dy/2, -dy/2, -dy/2,  dy/2], axis=1)
    z_c = np.stack([-dz/2, -dz/2, -dz/2, -dz/2,  dz/2,  dz/2,  dz/2,  dz/2], axis=1)

    local = np.stack([x_c, y_c, z_c], axis=2)  # (M, 8, 3)

    headings = boxes7[:, 6].astype(np.float32)
    for i in range(M):
        c = float(np.cos(headings[i]))
        s = float(np.sin(headings[i]))
        R = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        corners[i] = local[i] @ R.T

    corners += boxes7[:, :3].astype(np.float32).reshape(M, 1, 3)
    return corners


def mask_points_in_box(points_xyz: np.ndarray, box7: np.ndarray) -> np.ndarray:
    """
    Axis-aligned in the box's local frame.
    Args:
        points_xyz: (N, 3)
        box7: (7,) [x, y, z, dx, dy, dz, heading]
    Returns:
        mask: (N,) bool
    """
    cx, cy, cz, dx, dy, dz, heading = box7.astype(np.float32)

    pts = points_xyz.astype(np.float32) - np.array([cx, cy, cz], dtype=np.float32)[None, :]
    c = float(np.cos(-heading))
    s = float(np.sin(-heading))

    # rotate points into box local frame
    x_local = pts[:, 0] * c - pts[:, 1] * s
    y_local = pts[:, 0] * s + pts[:, 1] * c
    z_local = pts[:, 2]

    mask = (np.abs(x_local) <= dx / 2.0) & \
           (np.abs(y_local) <= dy / 2.0) & \
           (np.abs(z_local) <= dz / 2.0)
    return mask


class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training,
                         root_path=root_path, logger=logger)

        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / "ImageSets" / (self.split + ".txt")
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.infos = []
        self.seq_name_to_infos = self.include_waymo_data(self.mode)

    def include_waymo_data(self, mode):
        if self.logger is not None:
            self.logger.info("Loading Waymo dataset (for transform-like export)")

        waymo_infos = []
        seq_name_to_infos = {}

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / (f"{sequence_name}.pkl")
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue

            with open(info_path, "rb") as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

            seq_name_to_infos[infos[0]["point_cloud"]["lidar_sequence"]] = infos

        self.infos = waymo_infos
        if self.logger is not None:
            self.logger.info(f"Total skipped sequences: {num_skipped_infos}")
            self.logger.info(f"Total samples loaded: {len(self.infos)}")

        return seq_name_to_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ("%04d.npy" % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x,y,z,intensity,elongation,NLZ_flag,(maybe extra)]
        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]

        if not self.dataset_cfg.get("DISABLE_NLZ_FLAG_ON_POINTS", False):
            points_all = points_all[NLZ_flag == -1]

        # keep behavior, even though we export xyz only
        if self.dataset_cfg.get("POINTS_TANH_DIM", None) is None:
            points_all[:, 3] = np.tanh(points_all[:, 3])
        else:
            for dim_idx in self.dataset_cfg.POINTS_TANH_DIM:
                points_all[:, dim_idx] = np.tanh(points_all[:, dim_idx])

        return points_all  # (N,5)

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file: Path) -> Path:
        if not sequence_file.exists():
            found_sequence_file = sequence_file
            for pre_text in ["training", "validation", "testing"]:
                if not sequence_file.exists():
                    temp_sequence_file = Path(str(sequence_file).replace("segment", pre_text + "_segment"))
                    if temp_sequence_file.exists():
                        found_sequence_file = temp_sequence_file
                        break
            if not found_sequence_file.exists():
                found_sequence_file = Path(str(sequence_file).replace("_with_camera_labels", ""))
            if found_sequence_file.exists():
                sequence_file = found_sequence_file
        return sequence_file

    # ----------------------------------------------------------------------
    # Core function: produce transform.py-compatible dictionaries for one frame
    # ----------------------------------------------------------------------
    def generate_corrected_sequence_by_seq(
        self,
        sequence_name: str,
        cur_index: int,
        pre_frames: int = 70,
        post_frames: int = 70
    ):
        """
        transform.py-compatible output around (sequence_name, cur_index).

        Returns:
            final_points: dict { f"{i}_LIDAR_TOP": (4, Ni) float32 }
            final_poses:  dict { f"{i}_LIDAR_TOP": (4, 4) float32 }
            final_labels: dict { f"{i}_LIDAR_TOP": (Ni,) uint8 }
        """
        sequence_infos = self.seq_name_to_infos[sequence_name]
        seq_len = len(sequence_infos)

        # match transform.py window: i in [pre, post) with post exclusive
        start = max(0, cur_index - pre_frames)
        end_excl = min(seq_len, cur_index + post_frames)

        # Build a small seq_dict for just the window (like transform.py does per-log dict)
        seq_dict = {}
        for i in range(start, end_excl):
            info_i = sequence_infos[i]
            pose_i = info_i["pose"].reshape((4, 4)).astype(np.float32)

            # points: export xyz only, append ones, transpose => (4, N)
            pts_i = self.get_lidar(sequence_name, i)[:, :3].astype(np.float32)
            pts_i_h = np.hstack([pts_i, np.ones((pts_i.shape[0], 1), dtype=np.float32)]).T  # (4, N)

            # labels: default UNLABELED=1
            labels_i = np.full((pts_i.shape[0],), UNLABELED_ID, dtype=np.uint8)

            annos_i = info_i.get("annos", None)
            if annos_i is None:
                # no labels available: still output correct format
                seq_dict[i] = {
                    "points_all": pts_i_h,
                    "points_labels_all": labels_i,
                    "vehicle2global": pose_i,
                    "bboxes": np.zeros((0, 7), dtype=np.float32),
                    "bboxes_corner": np.zeros((0, 8, 3), dtype=np.float32),
                    "bboxes_id": np.array([], dtype=object),
                    "labels": np.array([], dtype=object),
                }
                continue

            boxes_i = annos_i["gt_boxes_lidar"].astype(np.float32)
            boxes_i = boxes_i[:, :7] if boxes_i.shape[1] > 7 else boxes_i
            corners_i = get_box_corners_np(boxes_i) if boxes_i.shape[0] > 0 else np.zeros((0, 8, 3), dtype=np.float32)

            seq_dict[i] = {
                "points_all": pts_i_h,                    # (4, N)
                "points_labels_all": labels_i,            # (N,)
                "vehicle2global": pose_i,                 # (4, 4)
                "bboxes": boxes_i,                        # (M, 7)
                "bboxes_corner": corners_i,               # (M, 8, 3)
                "bboxes_id": annos_i["obj_ids"],          # (M,)
                "labels": annos_i["name"],                # (M,)
            }

        # Prepare current frame references
        cur_ids = np.asarray(seq_dict[cur_index]["bboxes_id"])
        cur_pose = seq_dict[cur_index]["vehicle2global"]

        final_points = {}
        final_labels = {}
        final_poses = {}

        # Apply transform.py logic
        for i in range(start, end_excl):
            if i != cur_index:
                ids_i = np.asarray(seq_dict[i]["bboxes_id"])
                boxes_i = seq_dict[i]["bboxes"]
                corners_i = seq_dict[i]["bboxes_corner"]
                names_i = np.asarray(seq_dict[i]["labels"])

                j = 0
                # IMPORTANT: when we delete points, the arrays shrink; but boxes/ids loop stays on original boxes.
                # This is the same structure as transform.py: masks are re-computed on the current points array each time.
                while j < len(ids_i):
                    obj_id = ids_i[j]
                    box7 = boxes_i[j]

                    pts_xyz_now = seq_dict[i]["points_all"][:3].T  # (N_now, 3)
                    mask = mask_points_in_box(pts_xyz_now, box7)

                    if obj_id not in cur_ids:
                        # GHOST removal: delete masked points (same as transform.py)
                        if np.any(mask):
                            seq_dict[i]["points_all"] = np.delete(seq_dict[i]["points_all"], mask, axis=1)
                            seq_dict[i]["points_labels_all"] = np.delete(seq_dict[i]["points_labels_all"], mask)
                    else:
                        # Matched object: align to current frame box geometry, then map back into frame i coords
                        if np.any(mask):
                            idx_cur = int(np.where(cur_ids == obj_id)[0][0])

                            dst_box = np.copy(seq_dict[cur_index]["bboxes_corner"][idx_cur])  # (8,3)
                            src_box = np.copy(corners_i[j])                                  # (8,3)

                            c_dst = np.mean(dst_box, axis=0)
                            c_src = np.mean(src_box, axis=0)
                            dst0 = dst_box - c_dst
                            src0 = src_box - c_src

                            A = dst0.T @ src0
                            U, _, Vt = np.linalg.svd(A, full_matrices=False)
                            V = Vt.T
                            R = V @ U.T  # same as transform.py

                            # c = c_dst - c_src @ R
                            c = (c_dst - c_src @ R).reshape(3, 1).astype(np.float32)

                            curr_points = np.copy(seq_dict[i]["points_all"][:, mask])  # (4, n)
                            curr_points[:3] = (R.T @ curr_points[:3]) + c              # now in cur frame coords (still homogeneous)

                            src_pose = seq_dict[i]["vehicle2global"]
                            dst_pose = cur_pose

                            # new_points = inv(src_pose) @ dst_pose @ curr_points  (transform.py)
                            new_points = (np.linalg.inv(src_pose) @ dst_pose @ curr_points).astype(np.float32)

                            seq_dict[i]["points_all"][:, mask] = new_points

                            # label assignment (transform.py does this only in i != cur branch)
                            cls_name = str(names_i[j])
                            seq_dict[i]["points_labels_all"][mask] = np.uint8(LABEL_REMAP.get(cls_name, UNLABELED_ID))

                    j += 1

            key = f"{i}_LIDAR_TOP"
            final_points[key] = seq_dict[i]["points_all"].astype(np.float32)         # (4, N)
            final_labels[key] = seq_dict[i]["points_labels_all"].astype(np.uint8)    # (N,)
            final_poses[key] = seq_dict[i]["vehicle2global"].astype(np.float32)      # (4,4)

        return final_points, final_poses, final_labels

    # Convenience wrapper: original signature using global dataset index
    def generate_corrected_sequence(self, index, pre_frames=70, post_frames=70):
        info_cur = self.infos[index]
        seq_name = info_cur["point_cloud"]["lidar_sequence"]
        cur_idx = int(info_cur["point_cloud"]["sample_idx"])
        return self.generate_corrected_sequence_by_seq(seq_name, cur_idx, pre_frames, post_frames)


# -----------------------------
# Execution block (offline export)
# -----------------------------
if __name__ == "__main__":
    import argparse
    from pcdet.config import cfg, cfg_from_yaml_file

    parser = argparse.ArgumentParser("Waymo transform.py-style exporter")
    parser.add_argument('--cfg_file', type=str, default='cfgs/dataset_configs/waymo_dataset_dense_transform.yaml')
    parser.add_argument("--data_root", type=str, default="/", help="Path to data/waymo")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory (will create point_cloud/...)")
    parser.add_argument("--pre", type=int, default=70)
    parser.add_argument("--post", type=int, default=70)
    parser.add_argument("--process_all", action="store_true", help="Export every frame of every sequence")
    parser.add_argument("--sequence_name", type=str, default="", help="If set, export only this sequence_name")
    parser.add_argument("--frame_idx", type=int, default=-1, help="If set (>=0), export only this frame index in the sequence")
    parser.add_argument("--seq_batch_idx", type=int, default=-1, help="Batch index for sequence-level sharding (e.g., SLURM_ARRAY_TASK_ID). If >= 0, only a slice of sequences is processed.")
    parser.add_argument("--seq_batch_length", type=int, default=0, help="Number of sequences per batch. Required if --seq_batch_idx >= 0.")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    # Handle both "dataset config yaml" and "full model config yaml"
    dataset_cfg = cfg.DATA_CONFIG if hasattr(cfg, "DATA_CONFIG") else cfg

    logger = common_utils.create_logger()
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg,
        class_names=["Vehicle", "Pedestrian", "Cyclist"],
        training=True,
        root_path=Path(args.data_root),
        logger=logger
    )

    out_root = Path(args.save_dir) / "point_cloud"
    out_root.mkdir(parents=True, exist_ok=True)

    seq_names_all = [os.path.splitext(x)[0] for x in dataset.sample_sequence_list]

    # Recommended: only keep sequences that actually exist in seq_name_to_infos.
    # This avoids wasting batch slots on missing/failed-to-load sequences.
    seq_names = [s for s in seq_names_all if s in dataset.seq_name_to_infos]

    total_seqs = len(seq_names)

    batch_seq_names = None
    batch_start = None
    batch_end_excl = None

    if args.seq_batch_idx >= 0:
        if args.seq_batch_length <= 0:
            raise ValueError("--seq_batch_length must be > 0 when using --seq_batch_idx")

        batch_start = args.seq_batch_idx * args.seq_batch_length
        batch_end_excl = min(total_seqs, (args.seq_batch_idx + 1) * args.seq_batch_length)

        if batch_start >= total_seqs:
            logger.info(
                f"[Batch mode] seq_batch_idx={args.seq_batch_idx} start={batch_start} "
                f">= total_seqs={total_seqs}. Nothing to do; exiting."
            )
            raise SystemExit(0)

        batch_seq_names = seq_names[batch_start:batch_end_excl]

        logger.info(
            f"[Batch mode] seq_batch_idx={args.seq_batch_idx}, seq_batch_length={args.seq_batch_length} "
            f"=> processing sequences[{batch_start}:{batch_end_excl}] "
            f"({len(batch_seq_names)} sequences) out of {total_seqs} total available."
        )


    def export_one(sequence_name: str, k: int):
        final_points, final_poses, final_labels = dataset.generate_corrected_sequence_by_seq(
            sequence_name, k, pre_frames=args.pre, post_frames=args.post
        )

        seq_dir = out_root / sequence_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        points_path = seq_dir / f"{k:04d}_point"   # numpy will append .npz
        poses_path  = seq_dir / f"{k:04d}_pose"
        labels_path = seq_dir / f"{k:04d}_label"

        np.savez_compressed(points_path, **final_points)
        np.savez_compressed(poses_path,  **final_poses)
        np.savez_compressed(labels_path, **final_labels)

    if args.sequence_name:
        seq = args.sequence_name
        seq_infos = dataset.seq_name_to_infos[seq]
        print("seq_name, idx", args.sequence_name, args.frame_idx)
        for k in tqdm(range(args.frame_idx, len(seq_infos)), desc=f"Export {seq}"):
            export_one(seq, k)
    elif args.seq_batch_idx >= 0:
        # Process ONLY this batch’s sequences, export all frames in each sequence
        for seq in tqdm(batch_seq_names, desc=f"SeqBatch {args.seq_batch_idx} [{batch_start}:{batch_end_excl}]"):
            seq_infos = dataset.seq_name_to_infos[seq]
            for k in tqdm(range(len(seq_infos)), leave=False, desc=f"{seq}"):
                export_one(seq, k)
    elif args.process_all:
        for seq in tqdm(seq_names, desc="Sequences"):
            if seq not in dataset.seq_name_to_infos:
                continue
            seq_infos = dataset.seq_name_to_infos[seq]
            for k in tqdm(range(len(seq_infos)), leave=False, desc=f"{seq}"):
                export_one(seq, k)
    else:
        # minimal test: first sequence, first frame
        first_seq = next(iter(dataset.seq_name_to_infos.keys()))
        export_one(first_seq, 0)

    print(f"Done. Saved to: {out_root}")