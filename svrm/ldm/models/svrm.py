# Open Source Model Licensed under the Apache License Version 2.0 and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import time
import math
import cv2
import numpy as np
import itertools
import shutil
import mcubes
from tqdm import tqdm
import torch
import torch.nn.functional as F
from einops import rearrange
import trimesh

from ..modules.rendering_neus.mesh import Mesh
from ..modules.rendering_neus.rasterize import NVDiffRasterizerContext

from ..utils.ops import scale_tensor
from ..util import count_params, instantiate_from_config


class SVRMModel(torch.nn.Module):
    def __init__(
        self,
        img_encoder_config,
        img_to_triplane_config,
        render_config,
        device="cuda:0",
        **kwargs,
    ):
        super().__init__()

        self.img_encoder = instantiate_from_config(img_encoder_config).half()
        self.img_to_triplane_decoder = instantiate_from_config(
            img_to_triplane_config
        ).half()
        self.render = instantiate_from_config(render_config).half()
        self.device = device
        count_params(self, verbose=True)

    @torch.no_grad()
    def export_mesh_with_uv(
        self,
        data,
        mesh_size: int = 384,
    ):
        """
        color_type: 0 for ray texture, 1 for vertices texture
        """
        st = time.time()
        here = {"device": self.device, "dtype": torch.float16}
        input_view_image = data["input_view"].to(**here)  # [b, m, c, h, w]
        input_view_cam = data["input_view_cam"].to(**here)  # [b, m, 20]

        batch_size, input_view_num, *_ = input_view_image.shape
        assert batch_size == 1, "batch size should be 1"

        input_view_image = rearrange(input_view_image, "b m c h w -> (b m) c h w")
        input_view_cam = rearrange(input_view_cam, "b m d -> (b m) d")
        input_view_feat = self.img_encoder(input_view_image, input_view_cam)
        input_view_feat = rearrange(
            input_view_feat, "(b m) l d -> b (l m) d", m=input_view_num
        )

        # -- decoder
        torch.cuda.empty_cache()
        triplane_gen = self.img_to_triplane_decoder(
            input_view_feat
        )  # [b, 3, tri_dim, h, w]
        del input_view_feat
        torch.cuda.empty_cache()

        # --- triplane nerf render
        cur_triplane = triplane_gen[0:1]

        aabb = (
            torch.tensor([[-0.6, -0.6, -0.6], [0.6, 0.6, 0.6]]).unsqueeze(0).to(**here)
        )
        grid_out = self.render.forward_grid(
            planes=cur_triplane, grid_size=mesh_size, aabb=aabb
        )

        print(f"=====> LRM forward time: {time.time() - st}")
        st = time.time()

        # --- mesh extraction
        vtx, faces = mcubes.marching_cubes(
            0.0 - grid_out["sdf"].squeeze(0).squeeze(-1).cpu().float().numpy(), 0
        )
        bbox = aabb[0].cpu().numpy()
        vtx = vtx / (mesh_size - 1)
        vtx = vtx * (bbox[1] - bbox[0]) + bbox[0]
        vtx_colors = self.render.forward_points(
            cur_triplane, torch.tensor(vtx).unsqueeze(0).to(**here)
        )
        vtx_colors = vtx_colors["rgb"].float().squeeze(0).cpu().numpy()
        mesh = trimesh.Trimesh(
            vertices=vtx,
            faces=faces,
            vertex_colors=(vtx_colors * 255).clip(0, 255).astype(np.uint8),
        )

        print(f"=====> MESH extraction time: {time.time() - st}")
        st = time.time()

        return mesh
