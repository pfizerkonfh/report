#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer

ENABLE_DROP_GAUSSIAN = True
ENABLE_SH_DROP = True
TOTAL_ITER = 10000
WARMUP_ITER = 2000
MAX_DROP_RATE = 0.2
SH_DROP_MAX = 0.3


def render_fastgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, mult, scaling_modifier = 1.0, override_color = None, 
                  get_flag=None, metric_map = None, is_train=False, iteration=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map==None:
        metric_map=torch.zeros(int(viewpoint_camera.image_height)*int(viewpoint_camera.image_width), dtype=torch.int, device='cuda')

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult = mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map = metric_map
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # dc, shs = pc.get_features_dc, pc.get_features_rest

# 高阶SH更容易产生高频噪声与过拟合
#####################################################################################################################
            dc, shs = pc.get_features_dc, pc.get_features_rest

            # =========================================================
            # Frequency-aware SH Regularization
            # =========================================================

            if is_train and ENABLE_SH_DROP and iteration is not None:

                eps = 1e-6

                if iteration < WARMUP_ITER:
                    sh_drop_rate = 0.0
                else:
                    progress = (iteration - WARMUP_ITER) / (TOTAL_ITER - WARMUP_ITER)
                    progress = min(max(progress, 0.0), 1.0)

                    sh_drop_rate = SH_DROP_MAX * 0.5 * (
                            1 - math.cos(math.pi * progress)
                    )

                if sh_drop_rate > eps and shs is not None:

                    shs_reg = shs.clone()

                    total_coeffs = shs.shape[-1]

                    current_idx = 0

                    for degree in range(1, pc.active_sh_degree + 1):

                        coeff_num = 2 * degree + 1

                        start_idx = current_idx
                        end_idx = current_idx + coeff_num

                        # 防止越界
                        if end_idx > total_coeffs:
                            break

                        degree_ratio = degree / pc.active_sh_degree

                        degree_drop_rate = sh_drop_rate * degree_ratio
                        degree_drop_rate = min(degree_drop_rate, 0.5)

                        keep_prob = 1.0 - degree_drop_rate

                        # shape:
                        # [N,1,coeff_num]
                        mask = (
                                torch.rand(
                                    shs.shape[0],
                                    1,
                                    coeff_num,
                                    device=shs.device
                                ) < keep_prob
                        ).float()

                        # compensation
                        mask = mask / (keep_prob + eps)

                        # broadcast:
                        # [N,3,coeff_num]
                        shs_reg[:, :, start_idx:end_idx] = (
                                shs_reg[:, :, start_idx:end_idx]
                                * mask
                        )

                        current_idx = end_idx

                    shs = shs_reg
################################################################################################################
    else:
        colors_precomp = override_color


# drop的概率方面改进
################################################################################################################

    # # 原版DropGaussian
    # if is_train:
    #     # Create initial compensation factor (1 for each Gaussian)
    #     compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
    #
    #     # Apply DropGaussian with compensation
    #     drop_rate = 0.2 * (iteration/10000)
    #     d = torch.nn.Dropout(p=drop_rate)
    #     compensation = d(compensation)
    #
    #     # Apply to opacity
    #     opacity = opacity * compensation[:, None]



    # # 余弦退火动态 Drop Rate
    # if is_train and ENABLE_DROP_GAUSSIAN and iteration is not None:
    #     compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
    #     if iteration > WARMUP_ITER:
    #         # 余弦退火：先升后降
    #         progress = (iteration - WARMUP_ITER) / (TOTAL_ITER - WARMUP_ITER)
    #         drop_rate = MAX_DROP_RATE * 0.5 * (1 - math.cos(math.pi * progress))
    #     else:
    #         drop_rate = 0.0
    #     drop_rate = min(drop_rate, MAX_DROP_RATE)
    #     d = torch.nn.Dropout(p=drop_rate)
    #     compensation = d(compensation)
    #
    #     # Apply to opacity
    #     opacity = opacity * compensation[:, None]

    # 自适应重要性感知 DropGaussian
    # if is_train and ENABLE_DROP_GAUSSIAN and iteration is not None:
    #     if iteration > WARMUP_ITER:
    #         base_drop_rate = MAX_DROP_RATE * min(iteration / TOTAL_ITER, 1.0)
    #     else:
    #         base_drop_rate = 0.0
    #
    #     if base_drop_rate > 0:
    #         # 核心特征提取
    #         opacity_val = pc.get_opacity.squeeze()
    #         scale_val = pc.get_scaling.amax(dim=1)
    #         grad_val = pc.xyz_gradient_accum.norm(dim=1)
    #
    #         # 自适应重要性计算
    #         importance = torch.sigmoid(0.4 * opacity_val + 0.3 * scale_val + 0.3 * grad_val)
    #         adaptive_drop = base_drop_rate * (1.0 - importance)
    #
    #         # 向量化Drop
    #         keep_mask = torch.rand_like(opacity_val) > adaptive_drop
    #
    #         opacity_val *= keep_mask.float() / (1.0 - base_drop_rate)
    #
    #         pc.set_opacity = opacity_val.unsqueeze(-1)
    #
    #         del scale_val, grad_val, importance, adaptive_drop, keep_mask

    # 自适应重要性感知 DropGaussian
    if is_train and ENABLE_DROP_GAUSSIAN and iteration is not None:
        warmup_iter = WARMUP_ITER
        max_drop_rate = MAX_DROP_RATE
        total_iter = TOTAL_ITER
        eps = 1e-8
        if iteration < warmup_iter:
            base_drop_rate = 0.0
        else:
            progress = (iteration - warmup_iter) / (total_iter - warmup_iter)
            progress = min(max(progress, 0.0), 1.0)

            base_drop_rate = max_drop_rate * 0.5 * (
                    1 - math.cos(math.pi * progress)
            )
        base_drop_rate = min(base_drop_rate, max_drop_rate)

        if base_drop_rate > eps:
            # 提取特征
            opacity_val = pc.get_opacity.squeeze()
            scale_val = pc.get_scaling.amax(dim=1)
            grad_val = pc.xyz_gradient_accum.norm(dim=1)

            # 归一化
            scale_val = scale_val / (scale_val.max() + eps)
            grad_val = grad_val / (grad_val.max() + eps)

            # 可配置重要性权重
            w_op, w_scale, w_grad = 0.4, 0.3, 0.3
            importance = torch.sigmoid(w_op * opacity_val + w_scale * scale_val + w_grad * grad_val).detach()

            adaptive_drop_rate = base_drop_rate * (1.0 - importance)

            keep_mask = torch.rand_like(opacity_val) > adaptive_drop_rate

            compensation = keep_mask.float() / (1.0 - adaptive_drop_rate + eps)
            temp_opacity = opacity_val * compensation

            temp_opacity = torch.clamp(temp_opacity, 1e-6, 1.0)
            opacity = temp_opacity.unsqueeze(-1)

            del scale_val, grad_val, importance, adaptive_drop_rate, keep_mask, compensation

################################################################################################################

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1) # DropGaussian
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_metric_counts" : accum_metric_counts}