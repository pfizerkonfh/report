#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 3DGS 完整模型抽稀出轻量彩色点云 PLY（用于稳定、秒开的 3D 预览）。
输出: exp/viewer/points_light.ply  (x,y,z + uchar r,g,b)
"""
import os, struct, numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../exp
SRC = os.path.join(BASE, "point_cloud/iteration_10000/point_cloud.ply")
OUT = os.path.join(BASE, "viewer/points_light.ply")
SH_C0 = 0.28209479177387814
TARGET = 180000  # 目标点数


def parse_header(data: bytes):
    # 解析 ply 头，返回 (prop_names, vertex_count, data_offset)
    lines = []
    i = 0
    assert data[:3] == b"ply"
    # 头以 "end_header\n" 结束
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    props = []
    n = 0
    for ln in header:
        if ln.startswith("element vertex"):
            n = int(ln.split()[-1])
        elif ln.startswith("property"):
            # property float x
            props.append(ln.split()[-1])
    return props, n, end


def main():
    with open(SRC, "rb") as f:
        data = f.read()
    props, n, off = parse_header(data)
    # 所有属性均为 float32
    dt = np.dtype([(p, "<f4") for p in props])
    verts = np.frombuffer(data, dtype=dt, count=n, offset=off)
    xs = verts["x"]; ys = verts["y"]; zs = verts["z"]
    r = 0.5 + SH_C0 * verts["f_dc_0"]
    g = 0.5 + SH_C0 * verts["f_dc_1"]
    b = 0.5 + SH_C0 * verts["f_dc_2"]
    rgb = np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0)
    rgb_u = (rgb * 255.0 + 0.5).astype(np.uint8)

    # 抽稀：每隔 step 取一个点，使总数接近 TARGET
    step = max(1, int(np.ceil(n / TARGET)))
    idx = np.arange(0, n, step)
    px = xs[idx]; py = ys[idx]; pz = zs[idx]
    pr = rgb_u[idx]
    m = len(idx)
    print("源点数:", n, "抽稀后:", m, "step:", step)

    with open(OUT, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(("element vertex %d\n" % m).encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(b"end_header\n")
        buf = np.empty(m, dtype=np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
        buf["x"] = px; buf["y"] = py; buf["z"] = pz
        buf["red"] = pr[:, 0]; buf["green"] = pr[:, 1]; buf["blue"] = pr[:, 2]
        f.write(buf.tobytes())
    sz = os.path.getsize(OUT)
    print("已写出:", OUT, "(%.2f MB)" % (sz / 1e6))


if __name__ == "__main__":
    main()
