import argparse
import os
from pathlib import Path

import numpy as np

try:
    from plyfile import PlyData
except ImportError:  # fallback for environments without plyfile
    PlyData = None


SH_C0 = 0.282095


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def read_vertices(ply_path):
    if PlyData is not None:
        ply = PlyData.read(ply_path)
        if "vertex" not in ply:
            raise ValueError(f"Missing vertex data in {ply_path}")
        return ply["vertex"].data

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


def convert_one(ply_path, splat_path):
    v = read_vertices(ply_path)

    required = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    missing = [k for k in required if k not in v.dtype.names]
    if missing:
        raise ValueError(f"Missing fields in {ply_path}: {missing}")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(
        np.float32
    )
    scale = np.exp(scale)

    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(
        np.float32
    )
    quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.maximum(quat_norm, 1e-8)

    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    rgb = np.clip(SH_C0 * fdc + 0.5, 0.0, 1.0)
    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)

    alpha = sigmoid(v["opacity"].astype(np.float32))
    alpha_u8 = (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    rot_u8 = np.clip(quat * 128.0 + 128.0, 0, 255).astype(np.uint8)

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("sx", "<f4"),
            ("sy", "<f4"),
            ("sz", "<f4"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
            ("a", "u1"),
            ("rw", "u1"),
            ("rx", "u1"),
            ("ry", "u1"),
            ("rz", "u1"),
        ]
    )

    out = np.empty(xyz.shape[0], dtype=dtype)
    out["x"] = xyz[:, 0]
    out["y"] = xyz[:, 1]
    out["z"] = xyz[:, 2]
    out["sx"] = scale[:, 0]
    out["sy"] = scale[:, 1]
    out["sz"] = scale[:, 2]
    out["r"] = rgb_u8[:, 0]
    out["g"] = rgb_u8[:, 1]
    out["b"] = rgb_u8[:, 2]
    out["a"] = alpha_u8
    out["rw"] = rot_u8[:, 0]
    out["rx"] = rot_u8[:, 1]
    out["ry"] = rot_u8[:, 2]
    out["rz"] = rot_u8[:, 3]

    os.makedirs(os.path.dirname(splat_path), exist_ok=True)
    out.tofile(splat_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing .ply files exported from Gaussian splatting.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write .splat files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".ply")
    if not files:
        raise SystemExit(f"No .ply files found in {input_dir}")

    for ply_path in files:
        splat_path = output_dir / (ply_path.stem + ".splat")
        convert_one(str(ply_path), str(splat_path))
        print(f"Wrote {splat_path}")


if __name__ == "__main__":
    main()
