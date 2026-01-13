import argparse
import asyncio
import io
import json
import os
import pickle
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from cameras import MiniCam, look_at
from gs_renderer_4d import Renderer

try:
    import websockets
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: websockets. Install with `pip install websockets`."
    ) from exc


def _load_4d_model(save_name, out_dir):
    gaussians_path = os.path.join(out_dir, "gaussians", f"{save_name}_4d.pkl")
    motion_path = os.path.join(out_dir, "gaussians", f"{save_name}_4d_global_motion.pkl")
    if not os.path.exists(gaussians_path):
        raise FileNotFoundError(f"Missing 4D gaussians: {gaussians_path}")
    with open(gaussians_path, "rb") as f:
        gaussians_dict = pickle.load(f)
    motion = None
    if os.path.exists(motion_path):
        with open(motion_path, "rb") as f:
            motion = pickle.load(f)
    return gaussians_dict, motion


def _infer_T(motion, fallback_T):
    if motion and "translation" in motion:
        return int(motion["translation"].shape[0])
    return int(fallback_T)


def _encode_image(image, fmt, quality):
    buffer = io.BytesIO()
    if fmt == "png":
        Image.fromarray(image).save(buffer, format="PNG")
    else:
        Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


class CudaStreamServer:
    def __init__(
        self,
        opt,
        save_name,
        out_dir,
        fmt,
        quality,
        white_bg,
        transparent_bg,
    ):
        self.opt = opt
        self.save_name = save_name
        self.out_dir = out_dir
        if transparent_bg and fmt != "png":
            fmt = "png"
        self.format = fmt
        self.quality = quality
        self.white_bg = white_bg
        self.transparent_bg = transparent_bg

        gaussians_dict, motion = _load_4d_model(save_name, out_dir)
        self.motion = motion
        self.T = _infer_T(motion, opt.T)

        self.renderer = Renderer(T=self.T, sh_degree=opt.sh_degree, white_background=white_bg)
        self.renderer.initialize(gaussians_dict)

        if motion:
            device = torch.device("cuda")
            self.renderer.gaussian_translation = torch.tensor(
                motion["translation"], device=device
            )
            self.renderer.gaussian_scale = torch.tensor(
                motion["scale"], device=device
            )

    def render(self, payload):
        W = int(payload.get("width", self.opt.W))
        H = int(payload.get("height", self.opt.H))
        fov = float(payload.get("fov", self.opt.fovy))
        fovy = np.deg2rad(fov)
        fovx = 2 * np.arctan(np.tan(fovy / 2) * W / H)
        znear = float(payload.get("znear", self.opt.near))
        zfar = float(payload.get("zfar", self.opt.far))

        pos = np.array(payload.get("pos", [0, 0, self.opt.radius]), dtype=np.float32)
        target = np.array(payload.get("target", [0, 0, 0]), dtype=np.float32)
        time_raw = payload.get("time", 0)
        time_idx = int(np.clip(round(float(time_raw)), 0, self.T - 1))

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = look_at(pos[None], target[None], opengl=True)[0]
        c2w[:3, 3] = pos

        cam = MiniCam(
            c2w=c2w,
            width=W,
            height=H,
            fovy=fovy,
            fovx=fovx,
            znear=znear,
            zfar=zfar,
            time=time_idx,
        )

        bg_color = None
        if self.transparent_bg:
            bg_color = torch.zeros(3, device="cuda")
        with torch.no_grad():
            outputs = self.renderer.render(
                cam,
                direct_render=True,
                account_for_global_motion=self.motion is not None,
                bg_color=bg_color,
            )
        image = outputs["image"].clamp(0, 1)
        if self.transparent_bg:
            alpha = outputs["alpha"].clamp(0, 1)
            alpha_safe = torch.clamp(alpha, min=1e-6)
            rgb = torch.where(alpha_safe > 1e-6, image / alpha_safe, torch.zeros_like(image))
            rgba = torch.cat([rgb, alpha], dim=0)
            image = (rgba * 255.0).byte().permute(1, 2, 0).cpu().numpy()
        else:
            image = (image * 255.0).byte().permute(1, 2, 0).cpu().numpy()
        return _encode_image(image, self.format, self.quality)


async def handle_client(websocket, server):
    hello = {
        "type": "hello",
        "T": server.T,
        "width": server.opt.W,
        "height": server.opt.H,
        "fov": server.opt.fovy,
        "format": server.format,
    }
    await websocket.send(json.dumps(hello))

    async for message in websocket:
        if isinstance(message, bytes):
            continue
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "camera":
            continue

        start = time.perf_counter()
        try:
            frame = server.render(payload)
            await websocket.send(frame)
        except Exception as exc:
            err = {"type": "error", "message": str(exc)}
            await websocket.send(json.dumps(err))
        finally:
            _ = time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to 4d yaml config.")
    parser.add_argument("--save_name", required=True, help="Save name, e.g. dogs-jump_1.")
    parser.add_argument("--out_dir", default="./gaussians", help="Training output dir.")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port.")
    parser.add_argument("--format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality.")
    parser.add_argument("--white_bg", action="store_true", help="Use white background.")
    parser.add_argument(
        "--transparent_bg",
        action="store_true",
        help="Render with transparent background (PNG only).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this server requires a GPU.")

    opt = OmegaConf.load(args.config)
    server = CudaStreamServer(
        opt=opt,
        save_name=args.save_name,
        out_dir=args.out_dir,
        fmt=args.format,
        quality=args.quality,
        white_bg=args.white_bg,
        transparent_bg=args.transparent_bg,
    )

    async def _run():
        async with websockets.serve(
            lambda ws: handle_client(ws, server), "0.0.0.0", args.port, max_size=2**23
        ):
            print(f"CUDA stream WS listening on ws://localhost:{args.port}")
            await asyncio.Future()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
