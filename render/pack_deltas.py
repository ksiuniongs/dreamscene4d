import argparse
import json
import os
from pathlib import Path

import numpy as np

SH_C0 = 0.282095


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def read_vertices(ply_path):
    with open(ply_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in header: {ply_path}")
            header_lines.append(line.decode("ascii", errors="ignore").strip())
            if header_lines[-1] == "end_header":
                break
        header = header_lines
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"Only binary_little_endian supported: {ply_path}")

        vertex_count = None
        properties = []
        in_vertex = False
        for line in header:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                properties.append((parts[2], parts[1]))

        if vertex_count is None:
            raise ValueError(f"No vertex element in header: {ply_path}")

        type_map = {
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "uchar": "u1",
            "uint8": "u1",
            "char": "i1",
            "int8": "i1",
            "short": "<i2",
            "int16": "<i2",
            "ushort": "<u2",
            "uint16": "<u2",
            "int": "<i4",
            "int32": "<i4",
            "uint": "<u4",
            "uint32": "<u4",
        }

        dtype = []
        for name, ptype in properties:
            if ptype not in type_map:
                raise ValueError(f"Unsupported PLY type {ptype} in {ply_path}")
            dtype.append((name, type_map[ptype]))

        data = np.fromfile(f, dtype=np.dtype(dtype), count=vertex_count)
        return data


def load_frame(ply_path):
    v = read_vertices(ply_path)
    required = ["x", "y", "z"]
    missing = [k for k in required if k not in v.dtype.names]
    if missing:
        raise ValueError(f"Missing fields in {ply_path}: {missing}")
    pos = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    has_fdc = all(k in v.dtype.names for k in ("f_dc_0", "f_dc_1", "f_dc_2"))
    has_opacity = "opacity" in v.dtype.names
    if not has_fdc or not has_opacity:
        raise ValueError(f"Missing color/opacity in {ply_path}")

    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    color = np.clip(SH_C0 * fdc + 0.5, 0.0, 1.0)
    opacity = sigmoid(v["opacity"].astype(np.float32)).reshape(-1, 1)

    has_scales = all(k in v.dtype.names for k in ("scale_0", "scale_1", "scale_2"))
    if not has_scales:
        raise ValueError(f"Missing scale in {ply_path}")
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(
        np.float32
    )
    scale = np.exp(scale)

    rgba = np.concatenate([color, opacity], axis=1).astype(np.float32)
    # Optional: load f_rest for SH1 (9 coeffs per channel)
    f_rest = None
    f_rest_fields = [f"f_rest_{i}" for i in range(27)]
    if all(name in v.dtype.names for name in f_rest_fields):
        f_rest = np.stack([v[name] for name in f_rest_fields], axis=1).astype(np.float32)
    return pos, rgba, scale, f_rest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing frame_*.ply sequence.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for packed buffers.",
    )
    parser.add_argument("--fps", type=float, default=12.0, help="Playback FPS.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".ply")
    if not files:
        raise SystemExit(f"No .ply files found in {input_dir}")

    pos0, rgba0, scale0, f_rest0 = load_frame(files[0])
    n_points = pos0.shape[0]
    k_frames = len(files)

    delta = np.zeros((k_frames, n_points, 4), dtype=np.float32)
    delta[:, :, 3] = 0.0
    rgba = np.zeros((k_frames, n_points, 4), dtype=np.float32)
    scale = np.zeros((k_frames, n_points, 4), dtype=np.float32)
    scale[:, :, 3] = 0.0
    f_rest = None
    if f_rest0 is not None:
        f_rest = np.zeros((k_frames, n_points, 27), dtype=np.float32)

    for i, ply_path in enumerate(files):
        pos, frgba, sc, frest = load_frame(ply_path)
        if pos.shape[0] != n_points:
            raise ValueError(f"Point count mismatch in {ply_path}")
        delta[i, :, :3] = pos - pos0
        rgba[i, :, :4] = frgba
        scale[i, :, :3] = sc
        if f_rest is not None:
            if frest is None:
                raise ValueError(f"Missing f_rest in {ply_path}")
            f_rest[i, :, :] = frest
        print(f"Packed {ply_path.name}")

    os.makedirs(output_dir, exist_ok=True)
    pos0_path = output_dir / "pos0.bin"
    delta_path = output_dir / "delta.bin"
    rgba_path = output_dir / "rgba.bin"
    scale_path = output_dir / "scale.bin"
    f_rest_path = output_dir / "frest.bin"
    meta_path = output_dir / "meta.json"

    pos0.tofile(pos0_path)
    delta.tofile(delta_path)
    rgba.tofile(rgba_path)
    scale.tofile(scale_path)
    if f_rest is not None:
        f_rest.tofile(f_rest_path)

    meta = {
        "width": n_points,
        "height": k_frames,
        "points": n_points,
        "frames": k_frames,
        "fps": args.fps,
        "rgba": "rgba.bin",
        "scale": "scale.bin",
        "frest": "frest.bin" if f_rest is not None else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {pos0_path}")
    print(f"Wrote {delta_path}")
    print(f"Wrote {rgba_path}")
    print(f"Wrote {scale_path}")
    if f_rest is not None:
        print(f"Wrote {f_rest_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
