# # v2: sparse convolution by stacking the time, add skip connection and multi-scale feature
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import numpy as np
import spconv.pytorch as spconv
from functools import partial


# ----------------------------
# Dense conv helpers (unchanged decoder & fusion)
# ----------------------------
def conv3x3(in_channels, out_channels, bias=False):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
    )

def deconv3x3(in_channels, out_channels, stride):
    return nn.ConvTranspose3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        output_padding=1,
        bias=False,
    )

def maxpool2x2(stride):
    return nn.MaxPool3d(kernel_size=2, stride=stride, padding=0)

def relu(inplace=True):
    return nn.ReLU(inplace=inplace)

def bn(num_features):
    return nn.BatchNorm3d(num_features=num_features)


class SoftIoUWithSigmoidLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum()
        union = (pred + target - pred * target).sum().clamp(min=1e-6)
        return (1.0 - inter / union).mean()


class ConvBlock(nn.Module):
    """
    Matches your dense encoder's block used for the final fusion (block5).
    """
    def __init__(self, num_layer, in_channels, out_channels, max_pool=False):
        super().__init__()
        layers = []
        for i in range(num_layer):
            _in = in_channels if i == 0 else out_channels
            layers += [conv3x3(_in, out_channels), bn(out_channels), relu()]
        if max_pool:
            layers.append(maxpool2x2(stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ----------------------------
# spconv building blocks
# ----------------------------
def _post_act_block(in_channels, out_channels, kernel_size, indice_key=None,
                    stride=1, padding=0, conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size,
                                 padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                          indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    return spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(inplace=True),
    )


# ----------------------------
# Sparse encoder that mirrors the dense encoder's spatial schedule
# ----------------------------
class SparseEncoderMirror(nn.Module):
    """
    Input dict (dataset format):
        voxel_features: [N, C_in] float32 (C_in = p_pre time channels)
        voxel_coords:   [N, 4] int32  (b, z, y, x)
        batch_size:     int
    Spatial schedule (L, W, H) = (512, 256, 32):
        c1: (B,  32, 256, 128, 16)
        c2: (B,  64, 128,  64,  8)
        c3: (B, 128,  64,  32,  4)
        c4: (B, 256,  64,  32,  4)
        fuse -> c5: (B, 256, 128,  64,  8)
    Returns:
        c5_lwh (dense), sp_raw (the input SparseConvTensor at full res)
    """
    def __init__(self, in_channels, num_layers, num_filters):
        super().__init__()
        assert num_layers == [2, 2, 3, 6, 5]
        assert num_filters == [32, 64, 128, 256, 256]

        self.in_channels = in_channels
        self.out_channels = num_filters[4]
        self.sparse_shape = (32, 256, 512)  # (Z, Y, X)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # stem to 32ch @ full res
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(inplace=True),
        )
        # stage1 (stay), then down x2
        self.stage1 = spconv.SparseSequential(
            _post_act_block(num_filters[0], num_filters[0], 3, padding=1, indice_key='subm1',
                            conv_type='subm', norm_fn=norm_fn)  # total 2 layers incl. stem
        )
        self.down1 = _post_act_block(num_filters[0], num_filters[0], 3, stride=2, padding=1,
                                     indice_key='spconv2', conv_type='spconv', norm_fn=norm_fn)
        # stage2 (64ch), then down x2
        self.stage2 = spconv.SparseSequential(
            _post_act_block(num_filters[0], num_filters[1], 3, padding=1, indice_key='subm2',
                            conv_type='subm', norm_fn=norm_fn),
            _post_act_block(num_filters[1], num_filters[1], 3, padding=1, indice_key='subm2',
                            conv_type='subm', norm_fn=norm_fn),
        )
        self.down2 = _post_act_block(num_filters[1], num_filters[1], 3, stride=2, padding=1,
                                     indice_key='spconv3', conv_type='spconv', norm_fn=norm_fn)
        # stage3 (128ch), then down x2
        self.stage3 = spconv.SparseSequential(
            _post_act_block(num_filters[1], num_filters[2], 3, padding=1, indice_key='subm3',
                            conv_type='subm', norm_fn=norm_fn),
            _post_act_block(num_filters[2], num_filters[2], 3, padding=1, indice_key='subm3',
                            conv_type='subm', norm_fn=norm_fn),
            _post_act_block(num_filters[2], num_filters[2], 3, padding=1, indice_key='subm3',
                            conv_type='subm', norm_fn=norm_fn),
        )
        self.down3 = _post_act_block(num_filters[2], num_filters[2], 3, stride=2, padding=1,
                                     indice_key='spconv4', conv_type='spconv', norm_fn=norm_fn)
        # stage4 (256ch), no down
        self.stage4 = spconv.SparseSequential(
            _post_act_block(num_filters[2], num_filters[3], 3, padding=1, indice_key='subm4',
                            conv_type='subm', norm_fn=norm_fn),
            _post_act_block(num_filters[3], num_filters[3], 3, padding=1, indice_key='subm4',
                            conv_type='subm', norm_fn=norm_fn),
            _post_act_block(num_filters[3], num_filters[3], 3, padding=1, indice_key='subm4',
                            conv_type='subm', norm_fn=norm_fn),
            _post_act_block(num_filters[3], num_filters[3], 3, padding=1, indice_key='subm4',
                            conv_type='subm', norm_fn=norm_fn),
            _post_act_block(num_filters[3], num_filters[3], 3, padding=1, indice_key='subm4',
                            conv_type='subm', norm_fn=norm_fn),
            _post_act_block(num_filters[3], num_filters[3], 3, padding=1, indice_key='subm4',
                            conv_type='subm', norm_fn=norm_fn),
        )

        # dense fusion at (L/4, W/4, H/4) = (128, 64, 8)
        fuse_in = sum(num_filters[0:4])  # 32+64+128+256 = 480
        self.block5 = ConvBlock(num_layers[4], fuse_in, num_filters[4], max_pool=False)

    @staticmethod
    def _to_LWH(dense_zyx):
        # (B,C,Z,Y,X) -> (B,C,L=X,W=Y,H=Z)
        return dense_zyx.permute(0, 1, 4, 3, 2).contiguous()

    def _batch_to_sparse(self, batch):
        """
        Move batch sparse arrays to module device and build SparseConvTensor.
        """
        device = next(self.parameters()).device
        feats = batch['voxel_features']
        coors = batch['voxel_coords']
        B = int(batch['batch_size'])

        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        if isinstance(coors, np.ndarray):
            coors = torch.from_numpy(coors)

        feats = feats.to(device=device, dtype=torch.float32, non_blocking=True)
        coors = coors.to(device=device, dtype=torch.int32, non_blocking=True)

        # guard empty
        if feats.numel() == 0 or coors.numel() == 0:
            feats = torch.zeros((0, self.in_channels), device=device, dtype=torch.float32)
            coors = torch.zeros((0, 4), device=device, dtype=torch.int32)

        sp = spconv.SparseConvTensor(
            features=feats,
            indices=coors,
            spatial_shape=(32, 256, 512),  # (Z, Y, X)
            batch_size=B
        )
        return sp, B

    def forward(self, batch):
        """
        Args:
            batch: dict from dataset.collate_batch(...)
        Returns:
            c5_lwh: (B, 256, 128, 64, 8) dense
            sp_raw: input SparseConvTensor at full resolution (for skip head)
        """
        sp_raw, B = self._batch_to_sparse(batch)

        # Stage 1
        s1 = self.conv_input(sp_raw)   # 32ch @ full res
        s1 = self.stage1(s1)
        s1_down = self.down1(s1)       # -> (256,128,16)
        c1 = self._to_LWH(s1_down.dense())  # (B,32,256,128,16)

        # Stage 2
        s2 = self.stage2(s1_down)
        s2_down = self.down2(s2)       # -> (128,64,8)
        c2 = self._to_LWH(s2_down.dense())  # (B,64,128,64,8)

        # Stage 3
        s3 = self.stage3(s2_down)
        s3_down = self.down3(s3)       # -> (64,32,4)
        c3 = self._to_LWH(s3_down.dense())  # (B,128,64,32,4)

        # Stage 4
        s4 = self.stage4(s3_down)      # -> (64,32,4)
        c4 = self._to_LWH(s4.dense())  # (B,256,64,32,4)

        # Fuse to (128,64,8)
        _L, _W, _H = (512 // 4, 256 // 4, 32 // 4)  # (128, 64, 8)
        c1i = F.interpolate(c1, size=(_L, _W, _H), mode="trilinear", align_corners=True)
        c2i = F.interpolate(c2, size=(_L, _W, _H), mode="trilinear", align_corners=True)
        c3i = F.interpolate(c3, size=(_L, _W, _H), mode="trilinear", align_corners=True)
        c4i = F.interpolate(c4, size=(_L, _W, _H), mode="trilinear", align_corners=True)

        c_aggr = torch.cat((c1i, c2i, c3i, c4i), dim=1)  # 480ch
        c5 = self.block5(c_aggr)                          # (B,256,128,64,8)
        return c5, sp_raw


# ----------------------------
# Decoder (unchanged from dense baseline)
# ----------------------------
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            deconv3x3(in_channels, 128, stride=2),
            bn(128),
            relu(),
            conv3x3(128, 128),
            bn(128),
            relu(),
            deconv3x3(128, 64, stride=2),
            bn(64),
            relu(),
            conv3x3(64, 64),
            bn(64),
            relu(),
            conv3x3(64, out_channels, bias=True),
        )
    def forward(self, x):
        return self.block(x)


# ----------------------------
# Top-level model: dict in -> loss/logits out
# ----------------------------
class Conv3DSparseForecasting(nn.Module):
    """
    Expects the dataset dict as input:
        {
            'voxel_features': [N, p_pre],
            'voxel_coords':   [N, 4] (b,z,y,x),
            'labels':         [B, p_post, 512, 256, 32],
            'invalid':        [B, p_post, 512, 256, 32] (bool),
            'batch_size':     int
        }
    Train: returns loss
    Eval : returns logits (B, p_post, 512, 256, 32)
    """
    def __init__(self, p_pre, p_post, use_soft_iou=False):
        super().__init__()
        self.p_pre  = p_pre
        self.p_post = p_post

        # Sparse encoder with shape parity
        self.encoder = SparseEncoderMirror(
            in_channels=self.p_pre,
            num_layers=[2, 2, 3, 6, 5],
            num_filters=[32, 64, 128, 256, 256]
        )

        # Dense decoder (unchanged)
        self.decoder = Decoder(in_channels=256, out_channels=self.p_post)

        # Sparse skip/copy head at full res (operates on raw input sparse tensor)
        self.skip_head_sparse = spconv.SparseSequential(
            spconv.SparseConv3d(
                self.p_pre, self.p_post, kernel_size=3, stride=1, padding=1, bias=True,
                indice_key='skip_fullres_k3'
            )
        )

        # Losses
        self.bce = nn.BCEWithLogitsLoss()
        self.use_soft_iou = use_soft_iou
        if self.use_soft_iou:
            self.soft_iou = SoftIoUWithSigmoidLoss()

    @autocast(enabled=False)
    def forward(self, batch):
        # ---- sanity: channel alignment ----
        vf = batch['voxel_features']
        vf_ch = (vf.shape[1] if isinstance(vf, np.ndarray) else int(vf.size(1)))
        if vf_ch != self.p_pre:
            raise RuntimeError(
                f"[Conv3DSparseForecasting] voxel_features has {vf_ch} channels but model.p_pre={self.p_pre}. "
                f"Instantiate the model with p_pre == dataset.p_pre (history+current)."
            )

        # ---- Encoder ----
        c5, sp_raw = self.encoder(batch)         # c5: (B,256,128,64,8)

        # ---- Decoder to full res ----
        dec = self.decoder(c5)                   # (B,p_post,512,256,32) in (L,W,H)

        # ---- Sparse skip head on raw input (channels = p_pre) ----
        skip_sp = self.skip_head_sparse(sp_raw)          # sparse @ (Z,Y,X)
        skip_dense_zyx = skip_sp.dense()                 # (B,p_post, Z=32, Y=256, X=512)
        skip_dense_lwh = skip_dense_zyx.permute(0,1,4,3,2).contiguous()  # -> (B,p_post,512,256,32)

        logits = dec + skip_dense_lwh                     # same behavior as dense baseline

        # ---- Train / Eval branches ----
        if self.training:
            device = logits.device
            gt = batch['labels'].to(device=device).clone()
            gt[gt > 0] = 1
            gt = gt.to(dtype=torch.float32, device=device)

            invalid = batch['invalid'].to(device=device).bool()

            valid_logits = logits[~invalid]
            valid_gt     = gt[~invalid]

            bce_loss = self.bce(valid_logits, valid_gt)
            if self.use_soft_iou:
                soft_iou_loss = self.soft_iou(valid_logits, valid_gt)
                return 0.5 * bce_loss + 0.5 * soft_iou_loss
            return bce_loss

        return logits
