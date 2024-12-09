import os
import shutil
import warnings
import sys

from aiohttp import web
from server import PromptServer
from scipy.spatial import KDTree

warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
from folder_paths import (
    get_filename_list,
    get_full_path,
    get_save_image_path,
    get_output_directory,
)
from huggingface_hub import snapshot_download
import torch
import numpy as np
from rembg import remove, new_session
import mcubes
import cv2
from einops import rearrange
from PIL import Image, ImageSequence, ImageOps, ImageColor
import io
import base64
import ipywidgets as widgets
from IPython.display import display
import trimesh
import torch.nn.functional as F
from typing import Tuple
import folder_paths
from os import path
from .infer import Removebg, Image2Views, Views2Mesh

# 添加当前目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
sys.path.insert(0, path.dirname(__file__))

# 设置访问路径
base64_data_Array = {""}


@PromptServer.instance.routes.get("/get_obj_Base64")
async def get_obj(request):
    return web.json_response(base64_data_Array)


def barycentric_interpolate(v0, v1, v2, c0, c1, c2, p):
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-8:
        return (c0 + c1 + c2) / 3
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    interpolate_color = u * c0 + v * c1 + w * c2
    return np.clip(interpolate_color, 0, 255)


def is_point_in_triangle(p, v0, v1, v2):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(
        np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


class RemoveBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "session": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "u2net",
                        "lazy": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "make_six_views"
    CATEGORY = "EX"

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.worker_i2v = None

    def make_six_views(
        self,
        image,
        session,
    ):
        if not isinstance(image, Image.Image):
            if isinstance(image, torch.Tensor):
                print(f"Original tensor shape: {image.shape}")
                if image.dim() == 4 and image.shape[0] == 1:
                    image = image.squeeze(0)
                    print(f"Squeezed tensor shape: {image.shape}")
                if image.dim() == 3:
                    if image.shape[2] in [1, 3, 4]:
                        print(f"Image is in [H, W, C] format with shape: {image.shape}")
                        pass
                    elif image.shape[0] in [1, 3, 4]:
                        print(f"Image is in [C, H, W] format with shape: {image.shape}")
                        image = image.permute(1, 2, 0)
                        print(f"Transposed image shape: {image.shape}")
                    else:
                        raise ValueError(f"Unsupported image shape: {image.shape}")
                else:
                    raise ValueError(f"Unsupported image dimensions: {image.dim()}")
                image = image.cpu().numpy()
                print(
                    f"Converted to numpy array, shape: {image.shape}, dtype: {image.dtype}"
                )
                if image.dtype in [np.float32, np.float64]:
                    image = np.clip(image * 255, 0, 255).astype(np.uint8)
                    print(f"Scaled image to uint8, dtype: {image.dtype}")
                elif image.dtype == np.uint8:
                    print(f"Image is already uint8, dtype: {image.dtype}")
                else:
                    image = image.astype(np.uint8)
                    print(f"Converted image to uint8, dtype: {image.dtype}")
                if image.ndim != 3 or image.shape[2] not in [1, 3, 4]:
                    raise ValueError(
                        f"Invalid image shape after processing: {image.shape}"
                    )
                image = Image.fromarray(image)
                print(f"Converted to PIL Image, size: {image.size}, mode: {image.mode}")
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
        else:
            print(
                f"Input image is already a PIL Image, size: {image.size}, mode: {image.mode}"
            )
        rgba = remove(image, session=new_session(session))
        rgba_img_tensor = pil2tensor(rgba)
        return (rgba_img_tensor,)


class GenerateSixViews:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgba": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "step": ("INT", {"default": 50}),
                "use_lite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "sixImages",
        "originalImage",
    )
    FUNCTION = "make_six_views"
    CATEGORY = "EX"

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.worker_i2v = None

    def make_six_views(self, rgba, seed, step, use_lite):
        rgba = tensor2pil(rgba)
        if rgba.mode != "RGBA":
            raise ValueError("Input image is not in RGBA format.")
        if self.worker_i2v is None:
            self.worker_i2v = Image2Views(use_lite=use_lite, device=self.device)
        res_img, pils = self.worker_i2v(rgba, seed, step)
        res_multi_img_tensor = pil2tensor(res_img[0])
        use_image_tensor = pil2tensor(res_img[1])
        return (
            res_multi_img_tensor,
            use_image_tensor,
        )


class Hunyuan3DNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sixImages": ("IMAGE",),
                "originalImage": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "mesh_size": (
                    "INT",
                    {
                        "default": 500,
                        "min": 0,
                        "max": 1024,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "max_number_of_faces": (
                    "INT",
                    {
                        "default": 90000,
                        "min": 1000,
                        "max": 1000000,
                        "step": 100,
                        "display": "number",
                    },
                ),
                "use_lite": ("BOOLEAN", {"default": False}),
                "is_simplify": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "imgTo3D"
    CATEGORY = "EX"

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.worker_v23 = None

    @staticmethod
    def download_model_files(repo_id: str, local_dir: str):
        """
        从 Huggingface 仓库下载所有文件到指定的本地目录。

        :param repo_id: Huggingface 仓库的 ID，例如 "tencent/Hunyuan3D-1"
        :param local_dir: 本地目标目录，例如 "weights"
        """
        print(
            f"模型文件未找到，正在从 Huggingface 仓库 {repo_id} 下载所有文件到 {local_dir} ..."
        )
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                repo_type="model",
                ignore_patterns=["*.gitignore"],
            )
            print("模型文件下载完成。")
        except Exception as e:
            print(f"下载模型文件时出错: {e}")
            raise

    def imgTo3D(
        self,
        sixImages,
        originalImage,
        seed,
        mesh_size,
        max_number_of_faces,
        use_lite,
        is_simplify,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, "weights")
        svrm_dir = os.path.join(weights_dir, "svrm")
        weights_path = os.path.join(svrm_dir, "svrm.safetensors")

        # 检查模型权重文件是否存在
        if not os.path.exists(weights_path):
            print(f"未找到模型文件 {weights_path}，开始下载模型文件...")
            repo_id = "tencent/Hunyuan3D-1"  # Huggingface 仓库 ID
            self.download_model_files(repo_id=repo_id, local_dir=weights_dir)
            # 检查下载后是否存在模型文件
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"下载后仍未找到模型文件 {weights_path}。请检查仓库 ID 或网络连接。"
                )

        config_path = os.path.join(base_dir, "svrm/configs/svrm.yaml")
        weights_path = os.path.join(base_dir, "weights/svrm/svrm.safetensors")

        if self.worker_v23 is None:
            self.worker_v23 = Views2Mesh(
                config_path,
                weights_path,
                use_lite=use_lite,
                device=self.device,
            )

        # 从视图生成网格
        mesh = self.worker_v23(
            tensor2pil(sixImages),
            tensor2pil(originalImage),
            seed=seed,
            mesh_size=mesh_size,
        )
        mesh.apply_transform(
            np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )

        # 简化模型
        if is_simplify:
            simplified_mesh = mesh.simplify_quadric_decimation(
                face_count=max_number_of_faces
            )
            # 映射颜色：使用KD树找到新顶点的最近原始顶点
            tree = KDTree(mesh.vertices)
            _, closest_indices = tree.query(simplified_mesh.vertices)

            # 根据最近点的颜色重新赋值
            simplified_colors = mesh.visual.vertex_colors[closest_indices]
            simplified_mesh.visual.vertex_colors = simplified_colors

            mesh = simplified_mesh

        # gltf材质贴图
        # texture_image = mesh.visual.to_texture().material.image.copy()
        # texture_image_tensor = pil2tensor(
        #     texture_image.resize((1024, 1024), Image.BILINEAR)
        # )

        return (mesh,)


class TriMeshViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "saveName": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "mesh",
                    },
                ),
                "saveToOutputFolder": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"display": "DISPLAY"},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "display"
    CATEGORY = "EX"

    def display(self, mesh, saveName, saveToOutputFolder):
        saved = list()
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_image_path(saveName, get_output_directory())
        )

        filename_with_batch_num = filename.replace("%batch_num%", str(1))
        file = f"{filename_with_batch_num}_{counter:05}_.glb"
        buffer = io.BytesIO()
        mesh.export(buffer, "glb")
        buffer.seek(0)
        binary_data = buffer.getvalue()
        base64_data = base64.b64encode(binary_data).decode("utf-8")

        # 全局变量修改
        global base64_data_Array
        base64_data_Array = {"base64": {"file": file, "data": base64_data}}
        # 是否保存到输出目录
        if saveToOutputFolder:
            mesh.export(path.join(full_output_folder, file))
            saved.append({"filename": file, "type": "output", "subfolder": subfolder})
        return {"ui": {"mesh": file}}


class SquareImage:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": (
                    "INT",
                    {"default": 1024, "min": 8, "max": 8096, "step": 16},
                ),
                "upscale_method": (cls.upscale_methods,),
                "padding_color": ("COLOR", {"default": (255, 255, 255)}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Square_Image",)
    FUNCTION = "make_square"
    CATEGORY = "EX"

    def make_square(
        self, image, resolution, upscale_method, padding_color=(255, 255, 255)
    ):
        ret_images = []

        # 映射用户选择的缩放方法到 PIL 的 resample 方法
        resample_methods = {
            "nearest-exact": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "area": Image.BOX,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }

        resample = resample_methods.get(upscale_method, Image.NEAREST)

        for img_tensor in image:
            # 将张量转换为 PIL 图像
            pil_image = tensor2pil(img_tensor)

            # 获取原始尺寸
            original_width, original_height = pil_image.size

            # 判断是否已经是正方形
            if original_width == original_height:
                square_image = pil_image
            else:
                # 计算填充尺寸
                max_side = max(original_width, original_height)
                delta_w = max_side - original_width
                delta_h = max_side - original_height
                padding = (
                    delta_w // 2,
                    delta_h // 2,
                    delta_w - (delta_w // 2),
                    delta_h - (delta_h // 2),
                )

                # 填充图像
                square_image = ImageOps.expand(pil_image, padding, fill=padding_color)

            # 缩放到指定分辨率
            square_image = square_image.resize(
                (resolution, resolution), resample=resample
            )

            # 转换回张量
            tensor_img = pil2tensor(square_image)
            ret_images.append(tensor_img)

        # 将处理后的图像张量堆叠起来
        return (torch.cat(ret_images, dim=0),)


# Register the node
NODE_CLASS_MAPPINGS = {
    "Hunyuan3DNode": Hunyuan3DNode,
    "GenerateSixViews": GenerateSixViews,
    "SquareImage": SquareImage,
    "RemoveBackground": RemoveBackground,
    "TriMeshViewer": TriMeshViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3DNode": "EX_Hunyuan3DNode",
    "GenerateSixViews": "EX_GenerateSixViews",
    "SquareImage": "EX_SquareImage",
    "RemoveBackground": "EX_RemoveBackground",
    "TriMeshViewer": "EX_TriMeshViewer",
}
