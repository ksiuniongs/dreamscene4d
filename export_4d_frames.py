import argparse
import os
import pickle

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from plyfile import PlyData, PlyElement

from cameras import orbit_camera, OrbitCamera, MiniCam
from gs_renderer_4d import Renderer


def _as_tensor(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    return torch.tensor(value, device=device)


def _infer_video_len(global_motion_path, input_dir, default_T):
    if global_motion_path and os.path.exists(global_motion_path):
        with open(global_motion_path, "rb") as f:
            motion = pickle.load(f)
        translation = motion.get("translation")
        if translation is not None:
            return int(translation.shape[0])
    if input_dir:
        files = [p for p in os.listdir(input_dir) if not p.startswith(".")]
        files.sort()
        return len(files)
    return default_T


def _write_ply(model, path, xyz, scales, rotations, opacities):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    normals = np.zeros_like(xyz)
    f_dc = (
        model._features_dc.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    f_rest = (
        model._features_rest.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )

    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scales, rotations), axis=1
    )
    dtype_full = [(attribute, "f4") for attribute in model.construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the 4D config yaml.")
    parser.add_argument("--save_path", required=True, help="Save name used during training.")
    parser.add_argument("--out_dir", default="./gaussians", help="Training output directory.")
    parser.add_argument("--input", default=None, help="Input frame directory (optional).")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of prebaked frames to export.")
    parser.add_argument("--start", type=int, default=0, help="Start time index (inclusive).")
    parser.add_argument("--T", type=int, default=0, help="Fallback video length if not found.")
    parser.add_argument(
        "--format",
        choices=["png", "ply", "both"],
        default="ply",
        help="Export format for prebaked frames.",
    )
    args = parser.parse_args()

    opt = OmegaConf.load(args.config)

    gaussians_path = os.path.join(args.out_dir, "gaussians", f"{args.save_path}_4d.pkl")
    global_motion_path = os.path.join(
        args.out_dir, "gaussians", f"{args.save_path}_4d_global_motion.pkl"
    )
    if not os.path.exists(gaussians_path):
        raise FileNotFoundError(f"Missing 4D gaussians: {gaussians_path}")

    T = _infer_video_len(global_motion_path, args.input, args.T)
    if T <= 0:
        raise ValueError("Cannot infer video length; provide --input or --T.")

    with open(gaussians_path, "rb") as f:
        gaussians_dict = pickle.load(f)

    device = torch.device("cuda")
    renderer = Renderer(T=T, sh_degree=opt.sh_degree)
    renderer.initialize(gaussians_dict)

    has_global_motion = False
    if os.path.exists(global_motion_path):
        with open(global_motion_path, "rb") as f:
            motion = pickle.load(f)
        renderer.gaussian_translation = _as_tensor(motion["translation"], device)
        renderer.gaussian_scale = _as_tensor(motion["scale"], device)
        has_global_motion = True

    out_dir = os.path.join(args.out_dir, "prebaked_frames", args.save_path)
    os.makedirs(out_dir, exist_ok=True)

    times = np.linspace(args.start, T - 1, args.num_frames).round().astype(int)
    for i, t in enumerate(times):
        if args.format in ("png", "both"):
            cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
            pose = orbit_camera(opt.elevation, 0, opt.radius)
            cur_cam = MiniCam(
                pose,
                opt.W,
                opt.H,
                cam.fovy,
                cam.fovx,
                cam.near,
                cam.far,
                time=int(t),
            )
            with torch.no_grad():
                outputs = renderer.render(
                    cur_cam,
                    direct_render=True,
                    account_for_global_motion=True,
                )
            image = outputs["image"].detach().cpu().numpy().astype(np.float32)
            image = np.transpose(image, (1, 2, 0))
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(image).save(os.path.join(out_dir, f"frame_{i:03d}.png"))

        if args.format in ("ply", "both"):
            with torch.no_grad():
                xyz, rotations, scales, opacities = renderer.gaussians.get_deformed_everything(
                    int(t), T
                )
            if has_global_motion:
                scale_t = renderer.gaussian_scale[int(t)]
                translation_t = renderer.gaussian_translation[int(t)]
                xyz = xyz * scale_t + translation_t
                scales = scales * scale_t
            _write_ply(
                renderer.gaussians,
                os.path.join(out_dir, f"frame_{i:03d}.ply"),
                xyz.detach().cpu().numpy(),
                scales.detach().cpu().numpy(),
                rotations.detach().cpu().numpy(),
                opacities.detach().cpu().numpy(),
            )

    print(f"Saved {len(times)} frames to {out_dir}")


if __name__ == "__main__":
    main()
