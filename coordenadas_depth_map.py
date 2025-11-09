import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def generate_depth_and_pointcloud(
        image: str,
        width: int,
        height: int,
        save_dir: str = "./output_depth",
        encoder: str = "vits",
        focal_length: float = None):
    """
    Processa imagem -> Depth Map -> Point Cloud (x, y, z)

    Retorna:
        points_3d (np.ndarray): N x 3 -> coordenadas 3D
        depth_actual (np.ndarray): depth map relativo
        depth_vis_path (str): caminho do depth map salvo
    """

    os.makedirs(save_dir, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEVICE).eval()

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    h, w = img_rgb.shape[:2]

    transform = Compose([
        Resize(width=width, height=height, keep_aspect_ratio=True, ensure_multiple_of=14,resize_method='lower_bound',image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    img_tensor = transform({'image': img_rgb})['image']
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = model(img_tensor)  # (1, H, W)

    # Resize para original
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth_actual = depth.cpu().numpy()

    # ==== Salvar Depth Map ====
    depth_norm = (depth_actual - depth_actual.min()) / (depth_actual.max() - depth_actual.min()) * 255.0
    depth_vis = depth_norm.astype(np.uint8)
    depth_vis_path = os.path.join(save_dir, f"resultado_depth.png")
    cv2.imwrite(depth_vis_path, cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO))

    # ==== Converter para Pontos 3D ====
    if focal_length is None:
        focal_length = max(h, w) * 0.8

    cx, cy = w / 2, h / 2
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    x_norm = (u - cx) / focal_length
    y_norm = (v - cy) / focal_length

    x_3d = x_norm * depth_actual
    y_3d = y_norm * depth_actual
    z_3d = depth_actual

    points_3d = np.stack([x_3d, y_3d, z_3d], axis=-1).reshape(-1, 3)

    return points_3d


def get_3d_point(points, width, x, y):
    
    height = points.shape[0] // width
    xyz = points.reshape((height, width, 3))

    return xyz[y, x, :]