import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def depth_to_3d_points(depth_map, focal_length=None, cx=None, cy=None):
    """
    Convert depth map to 3D point cloud
    
    Args:
        depth_map: depth map in meters (H, W)
        focal_length: focal length in pixels. If None, estimated from image dimensions
        cx, cy: principal point coordinates. If None, set to image center
    
    Returns:
        points: array of 3D points (N, 3) where N = H * W
    """
    h, w = depth_map.shape
    
    if focal_length is None:
        focal_length = max(h, w) * 0.8
    
    if cx is None:
        cx = w / 2
    if cy is None:
        cy = h / 2
    
    # Create grid of pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert to normalized coordinates
    x_norm = (u - cx) / focal_length
    y_norm = (v - cy) / focal_length
    
    # Convert to 3D coordinates
    x_3d = x_norm * depth_map
    y_3d = y_norm * depth_map
    z_3d = depth_map
    
    # Reshape to point cloud
    points = np.stack([x_3d, y_3d, z_3d], axis=-1).reshape(-1, 3)
    
    return points


def save_point_cloud(points, filename, max_points=50000):
    """
    Save point cloud to PLY file
    
    Args:
        points: array of 3D points (N, 3)
        filename: output filename
        max_points: maximum number of points to save (for performance)
    """
    if len(points) > max_points:
        # Randomly sample points if there are too many
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    with open(filename, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        # Write points
        for point in points:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(point[0], point[1], point[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save-pointcloud', dest='save_pointcloud', action='store_true', help='save 3D point cloud')
    parser.add_argument('--focal-length', type=float, default=None, help='focal length in pixels')
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=544,
            height=544,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path)
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        
        # Normalize depth for visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_vis = depth_normalized.cpu().numpy().astype(np.uint8)
        
        # Convert to actual depth values (you might need to scale this appropriately)
        # Depth Anything outputs relative depth, so you may need to calibrate for your use case
        depth_actual = depth.cpu().numpy()
        
        if args.grayscale:
            depth_display = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
        else:
            depth_display = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        
        base_filename = os.path.basename(filename)
        name_only = base_filename[:base_filename.rfind('.')]
        
        # Save 3D points if requested
        if args.save_pointcloud:
            # Convert to 3D points
            points_3d = depth_to_3d_points(
                depth_actual, 
                focal_length=args.focal_length,
                cx=w/2, 
                cy=h/2
            )
            
            txt_filename = os.path.join(args.outdir, name_only + '_coordinates.txt')
            np.savetxt(txt_filename, points_3d, fmt='%.6f', header='x y z')
            print(f"Saved coordinates to {txt_filename}")

        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, name_only + '_depth.png'), depth_display)
        else:
            split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
            combined_results = cv2.hconcat([raw_image, split_region, depth_display])
            
            caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
            captions = ['Raw image', 'Depth Anything']
            segment_width = w + margin_width
            
            for i, caption in enumerate(captions):
                text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]
                text_x = int((segment_width * i) + (w - text_size[0]) / 2)
                cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)
            
            final_result = cv2.vconcat([caption_space, combined_results])
            
            cv2.imwrite(os.path.join(args.outdir, name_only + '_img_depth.png'), final_result)