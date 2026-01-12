import argparse
import os
import sys

import math
import numpy as np
import torch
from PIL import Image

repo_root = os.path.dirname(os.path.abspath(__file__))
build_root = os.path.join(repo_root, "diff-gaussian-rasterization", "build")
if os.path.isdir(build_root):
    for entry in os.listdir(build_root):
        if entry.startswith("lib."):
            sys.path.insert(0, os.path.join(build_root, entry))
sys.path.append(os.path.join(repo_root, "diff-gaussian-rasterization"))

from gs_renderer import Renderer


def safe_normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def look_at(campos, target, opengl=True):
    if not opengl:
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    return np.stack([right_vector, up_vector, forward_vector], axis=1)


def orbit_camera(elevation, azimuth, radius=1, target=None, is_degree=True):
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = -radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl=True)
    T[:3, 3] = campos
    return T


def get_projection_matrix(znear, zfar, fovx, fovy):
    tan_half_fovy = math.tan(fovy / 2)
    tan_half_fovx = math.tan(fovx / 2)
    P = torch.zeros(4, 4)
    P[0, 0] = 1 / tan_half_fovx
    P[1, 1] = 1 / tan_half_fovy
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        w2c = np.linalg.inv(c2w)
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            get_projection_matrix(self.znear, self.zfar, self.FoVx, self.FoVy)
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a .ply file.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--W", type=int, default=512, help="Image width.")
    parser.add_argument("--H", type=int, default=512, help="Image height.")
    parser.add_argument("--fovy", type=float, default=49.1, help="Camera fovy in degrees.")
    parser.add_argument("--radius", type=float, default=2.0, help="Camera radius.")
    parser.add_argument("--elevation", type=float, default=0.0, help="Camera elevation (deg).")
    parser.add_argument("--azimuth", type=float, default=0.0, help="Camera azimuth (deg).")
    parser.add_argument("--white_bg", action="store_true", help="Use white background.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; CUDA renderer requires GPU.")

    renderer = Renderer(sh_degree=0, white_background=args.white_bg)
    renderer.initialize(args.input)

    fovy = math.radians(args.fovy)
    fovx = 2 * math.atan(math.tan(fovy / 2) * args.W / args.H)
    pose = orbit_camera(args.elevation, args.azimuth, args.radius)
    cur_cam = MiniCam(
        pose,
        args.W,
        args.H,
        fovy,
        fovx,
        0.01,
        100.0,
    )

    with torch.no_grad():
        outputs = renderer.render(cur_cam, account_for_global_motion=False)
    image = outputs["image"].detach().cpu().numpy().astype(np.float32)
    image = np.transpose(image, (1, 2, 0))
    image = (image * 255.0).clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    Image.fromarray(image).save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
