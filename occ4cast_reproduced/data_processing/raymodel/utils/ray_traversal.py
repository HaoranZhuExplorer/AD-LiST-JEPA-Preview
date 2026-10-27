#modified ray_traversal.py with a minimal change (look for the comment # >>> FIX START). Fixed original code index out of bound error.

from typing import List
import numpy as np
import torch
from cumm.inliner import NVRTCInlineBuilder
from cumm.common import TensorViewNVRTCKernel, TensorViewArrayLinalg

INLINER = NVRTCInlineBuilder([TensorViewNVRTCKernel, TensorViewArrayLinalg])
NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1
FREE_LABEL = 0

BINARY_OBSERVED = 1
BINARY_NOT_OBSERVED = 0

# DO NOT CHANGE
FLT_MAX = 1e9
RAY_STOP_DISTANCE_VOXEL = 1
DISTANCE_THESHOLD_IGNORE = 1.0
RAY_ROAD_IGNORE_DISTANCE = 1.0
VIS = False


def to_stride(shape: np.ndarray):
    stride = np.ones_like(shape)
    stride[: shape.shape[0] - 1] = np.cumprod(shape[::-1])[::-1][1:]
    return stride


def _indice_to_scalar(indices: torch.Tensor, shape: List[int]):
    assert indices.shape[1] == len(shape)
    stride = to_stride(np.array(shape, dtype=np.int64))
    scalar_inds = indices[:, -1].clone()
    for i in range(len(shape) - 1):
        scalar_inds += stride[i] * indices[:, i]
    return scalar_inds.contiguous()


def ray_traversal(
    points_origin,
    points,
    points_label,
    point_cloud_range,
    voxel_size,
    spatial_shape,
    free_n_theshold=0,
    occupied_n_theshold=1,
):
    """
    inputs:
        points_origin: [N, 3], x/y/z order, in global coordinate
        points: [N, 3+], x/y/z order, in global coordinate
        points_label: [N, ], labels
        point_cloud_range: list[6], xmin/ymin/zmin/xmax/ymax/zmax, in global coordinate
        voxel_size: list[3], x/y/z order
        spatial_shape: list[3], x/y/z order
        free_n_theshold: traversal num exceed this as free
    outputs:
        voxel_coors: [H,W,Z,3]
        voxel_state: [H,W,Z], -1/0/1 for not observed/free/occupied
        voxel_label: [H,W,Z], from point label
        voxel_occ_count: [H,W,Z], point num in this voxel
        voxel_free_count: [H,W,Z], free traversal in this voxel
    """
    _device = points.device
    voxel_size_numpy = np.asarray(voxel_size)
    point_cloud_range_numpy = np.asarray(point_cloud_range)
    spatial_shape_numpy = np.asarray(spatial_shape)

    voxel_size_device = torch.tensor(voxel_size).to(_device)
    point_cloud_range_device = torch.tensor(point_cloud_range).to(_device)

    # Filter points not in point_cloud_range (KEEP AS-IS for output consistency)
    inrange_x = torch.logical_and(points[:, 0] > point_cloud_range[0], points[:, 0] < point_cloud_range[3])
    inrange_y = torch.logical_and(points[:, 1] > point_cloud_range[1], points[:, 1] < point_cloud_range[4])
    inrange_z = torch.logical_and(points[:, 2] > point_cloud_range[2], points[:, 2] < point_cloud_range[5])
    inrange = torch.logical_and(inrange_x, torch.logical_and(inrange_y, inrange_z))

    points_inrange = points[inrange]
    points_origin_inrange = points_origin[inrange]
    points_labels_inrange = points_label[inrange]

    # voxel traversal
    voxel_occ_count = torch.full(spatial_shape, fill_value=0, device=_device, dtype=torch.int)
    voxel_free_count = torch.full(spatial_shape, fill_value=0, device=_device, dtype=torch.int)

    ray_start = points_inrange.contiguous()
    ray_end = points_origin_inrange[:, :3].contiguous()

    ray_start_stride = ray_start.stride()
    ray_end_stride = ray_end.stride()
    voxel_stride = voxel_occ_count.stride()

    assert ray_start.shape == ray_end.shape

    # ray start as points, ray end as origin
    # (KEEP AS-IS)
    assert torch.all(torch.logical_and(ray_start[:, 0] >= point_cloud_range_device[0],
                                       ray_start[:, 0] <= point_cloud_range_device[3]))
    assert torch.all(torch.logical_and(ray_start[:, 1] >= point_cloud_range_device[1],
                                       ray_start[:, 1] <= point_cloud_range_device[4]))
    assert torch.all(torch.logical_and(ray_start[:, 2] >= point_cloud_range_device[2],
                                       ray_start[:, 2] <= point_cloud_range_device[5]))

    INLINER.kernel_1d("voxel_traversal", ray_start.shape[0], 0, f"""
    auto ray_start_p = $ray_start + i*$(ray_start_stride[0]);
    auto ray_end_p = $ray_end + i*$(ray_end_stride[0]);

    // Bring the ray_start_p and ray_end_p in voxel coordinates
    float new_ray_start[3];
    float new_ray_end[3];
    float voxel_size_[3];
    new_ray_start[0] = ray_start_p[0] - $(point_cloud_range[0]);
    new_ray_start[1] = ray_start_p[1] - $(point_cloud_range[1]);
    new_ray_start[2] = ray_start_p[2] - $(point_cloud_range[2]);
    new_ray_end[0] = ray_end_p[0] - $(point_cloud_range[0]);
    new_ray_end[1] = ray_end_p[1] - $(point_cloud_range[1]);
    new_ray_end[2] = ray_end_p[2] - $(point_cloud_range[2]);
    voxel_size_[0] = $(voxel_size[0]);
    voxel_size_[1] = $(voxel_size[1]);
    voxel_size_[2] = $(voxel_size[2]);

    // Declare some variables that we will need
    float ray[3]; // keep the ray
    int step[3];
    float tDelta[3];
    int current_voxel[3];
    int last_voxel[3];
    int target_voxel[3];
    float _EPS = 1e-9;
    for(int k=0; k<3; k++) {{
        // Compute the ray
        ray[k] = new_ray_end[k] - new_ray_start[k];

        // Get the step along each axis
        step[k] = (ray[k] >=0) ? 1:-1;

        // Compute how much we need to move in t for the ray to move bin_size
        tDelta[k] = (ray[k] !=0) ? (step[k] * voxel_size_[k]) / ray[k]: {FLT_MAX};

        // Move the start and end points just a bit so that they are never on the boundary
        new_ray_start[k] = new_ray_start[k] + step[k]*voxel_size_[k]*_EPS;
        new_ray_end[k] = new_ray_end[k] - step[k]*voxel_size_[k]*_EPS;

        // Compute the first and the last voxels for the voxel traversal
        current_voxel[k] = (int) floor(new_ray_start[k] / voxel_size_[k]);
        last_voxel[k] = (int) floor(new_ray_end[k] / voxel_size_[k]);
        target_voxel[k] = (int) floor(new_ray_start[k] / voxel_size_[k]); // ray start as point, ray end as origin
    }}

    // Compute the values of t where the ray crosses the next boundaries
    float tMax[3];
    float current_coordinate;
    for (int k=0; k<3; k++) {{
        if (ray[k] !=0 ) {{
            current_coordinate = current_voxel[k]*voxel_size_[k];
            if (step[k] < 0 && current_coordinate < new_ray_start[k]) {{
                tMax[k] = current_coordinate;
            }}
            else {{
                tMax[k] = current_coordinate + step[k]*voxel_size_[k];
            }}
            tMax[k] = (tMax[k] - new_ray_start[k]) / ray[k];
        }}
        else {{
            tMax[k] = {FLT_MAX};
        }}
    }}

    // record point, +1 (already guarded; KEEP AS-IS)
    if (
        ((target_voxel[0] >= 0 && target_voxel[0] < $(spatial_shape[0])) &&
        (target_voxel[1] >= 0 && target_voxel[1] < $(spatial_shape[1])) &&
        (target_voxel[2] >= 0 && target_voxel[2] < $(spatial_shape[2])))
    ) {{
        auto targetIdx = target_voxel[0] * $(voxel_stride[0]) + target_voxel[1] * $(voxel_stride[1]) + target_voxel[2] * $(voxel_stride[2]);
        atomicAdd($voxel_occ_count + targetIdx, 1);
    }}

    // Start the traversal
    while(step[0]*(current_voxel[0] - last_voxel[0]) < {RAY_STOP_DISTANCE_VOXEL} &&
          step[1]*(current_voxel[1] - last_voxel[1]) < {RAY_STOP_DISTANCE_VOXEL} &&
          step[2]*(current_voxel[2] - last_voxel[2]) < {RAY_STOP_DISTANCE_VOXEL}){{

        if (tMax[0] < tMax[1]) {{
            if (tMax[0] < tMax[2]) {{
                current_voxel[0] = current_voxel[0] + step[0];
                if (current_voxel[0] < 0 || current_voxel[0] >= $(spatial_shape[0]))
                    break;
                tMax[0] = tMax[0] + tDelta[0];
            }}
            else {{
                current_voxel[2] = current_voxel[2] + step[2];
                if (current_voxel[2] < 0 || current_voxel[2] >= $(spatial_shape[2]))
                    break;
                tMax[2] = tMax[2] + tDelta[2];
            }}
        }}
        else {{
            if (tMax[1] < tMax[2]) {{
                current_voxel[1] = current_voxel[1] + step[1];
                if (current_voxel[1] < 0 || current_voxel[1] >= $(spatial_shape[1]))
                    break;
                tMax[1] = tMax[1] + tDelta[1];
            }}
            else {{
                current_voxel[2] = current_voxel[2] + step[2];
                if (current_voxel[2] < 0 || current_voxel[2] >= $(spatial_shape[2]))
                    break;
                tMax[2] = tMax[2] + tDelta[2];
            }}
        }}

        // >>> FIX START: guard the free-count atomicAdd with a FULL 3D bounds check.
        // This does NOT change results when current_voxel is in-bounds (the normal case),
        // but prevents illegal memory access when a rare boundary/epsilon/floor case produces OOB indices.
        if (
            ((current_voxel[0] >= 0 && current_voxel[0] < $(spatial_shape[0])) &&
             (current_voxel[1] >= 0 && current_voxel[1] < $(spatial_shape[1])) &&
             (current_voxel[2] >= 0 && current_voxel[2] < $(spatial_shape[2])))
        ) {{
            auto currentIdx = current_voxel[0] * $(voxel_stride[0]) + current_voxel[1] * $(voxel_stride[1]) + current_voxel[2] * $(voxel_stride[2]);

            auto distance2start =
                abs(current_voxel[0] - target_voxel[0]) * voxel_size_[0] +
                abs(current_voxel[1] - target_voxel[1]) * voxel_size_[1] +
                abs(current_voxel[2] - target_voxel[2]) * voxel_size_[2];

            auto distance2end =
                abs(current_voxel[0] - last_voxel[0]) * voxel_size_[0] +
                abs(current_voxel[1] - last_voxel[1]) * voxel_size_[1] +
                abs(current_voxel[2] - last_voxel[2]) * voxel_size_[2];

            auto distance2start_height = abs(current_voxel[2] - target_voxel[2]) * voxel_size_[2];

            if(distance2start > {DISTANCE_THESHOLD_IGNORE} && distance2end > {DISTANCE_THESHOLD_IGNORE}){{
                atomicAdd($voxel_free_count + currentIdx, 1);
            }}
        }}
        // <<< FIX END
    }}
    """, verbose_path="build/nvrtc/voxel_traversal")

    torch.cuda.synchronize()

    # set default not observed
    voxel_state = torch.full(spatial_shape, fill_value=NOT_OBSERVED, device=_device, dtype=torch.long)
    voxel_label = torch.full(spatial_shape, fill_value=FREE_LABEL, device=_device, dtype=torch.long)
    voxel_label_squeeze = voxel_label.reshape(-1)

    # set voxel label
    pcds_voxel = torch.div(
        points_inrange - point_cloud_range_device[:3],
        voxel_size_device,
        rounding_mode='floor'
    ).long()
    inds = _indice_to_scalar(pcds_voxel, voxel_state.shape)
    voxel_label_squeeze[inds] = points_labels_inrange

    # set free voxel
    voxel_state[voxel_free_count > free_n_theshold] = FREE
    # set occupied
    voxel_state[voxel_occ_count > occupied_n_theshold] = OCCUPIED

    xx = torch.arange(0, voxel_state.shape[0]).to(_device)
    yy = torch.arange(0, voxel_state.shape[1]).to(_device)
    zz = torch.arange(0, voxel_state.shape[2]).to(_device)
    grid_x, grid_y, grid_z = torch.meshgrid(xx, yy, zz, indexing='ij')
    voxel_coors = torch.stack([grid_x, grid_y, grid_z], axis=-1)

    return voxel_coors, voxel_state, voxel_label, voxel_occ_count, voxel_free_count
