#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计算 3DGS 渲染图与真值(GT)的 PSNR，并生成展示页所需的 gallery.json / metrics.json。
输出目录: exp/viewer/
"""
import os, json, math
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../exp
VIEWER = os.path.join(BASE, "viewer")
TEST_R = os.path.join(BASE, "test/ours_10000/renders")
TEST_G = os.path.join(BASE, "test/ours_10000/gt")
TRAIN_R = os.path.join(BASE, "train/ours_10000/renders")
TRAIN_G = os.path.join(BASE, "train/ours_10000/gt")
CAMERAS = os.path.join(BASE, "cameras.json")


def psnr_np(r: np.ndarray, g: np.ndarray) -> float:
    r = r.astype(np.float64) / 255.0
    g = g.astype(np.float64) / 255.0
    mse = np.mean((r - g) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def evaluate(render_dir, gt_dir, step=None):
    files = sorted(os.listdir(render_dir))
    if step:
        files = files[::step]
    out = []
    for f in files:
        r = np.asarray(Image.open(os.path.join(render_dir, f)).convert("RGB"))
        g = np.asarray(Image.open(os.path.join(gt_dir, f)).convert("RGB"))
        if r.shape != g.shape:
            g = np.asarray(Image.open(os.path.join(gt_dir, f)).convert("RGB").resize(r.shape[1::-1]))
        p = psnr_np(r, g)
        rel_r = "/" + os.path.relpath(os.path.join(render_dir, f), BASE).replace(os.sep, "/")
        rel_g = "/" + os.path.relpath(os.path.join(gt_dir, f), BASE).replace(os.sep, "/")
        out.append({"name": f, "render": rel_r, "gt": rel_g, "psnr": round(p, 2)})
    return out


def rel(p):
    return "/" + os.path.relpath(p, BASE).replace(os.sep, "/")


test = evaluate(TEST_R, TEST_G)
train = evaluate(TRAIN_R, TRAIN_G, step=8)

mean_test = round(sum(x["psnr"] for x in test) / len(test), 2)
mean_train = round(sum(x["psnr"] for x in train) / len(train), 2)

# 相机位姿数量与重建中心(用于初始化查看器视角)
with open(CAMERAS, "r", encoding="utf-8") as fp:
    cams = json.load(fp)
n_cam = len(cams)
centroid = [round(sum(c["position"][i] for c in cams) / n_cam, 4) for i in range(3)]
# 取中间相机作为初始机位
mid = cams[n_cam // 2]
init_pos = [round(v, 4) for v in mid["position"]]

gallery = {"test": test, "train": train}
with open(os.path.join(VIEWER, "gallery.json"), "w", encoding="utf-8") as fp:
    json.dump(gallery, fp, ensure_ascii=False, indent=2)

metrics = {
    "scene": "实验室场景 (LLFF 前向采集)",
    "model_format": "3D Gaussian Splatting (.ply, sh_degree=3)",
    "gaussian_count_10000": 387299,
    "gaussian_count_7000": 375387,
    "iterations": 10000,
    "test_views": len(test),
    "train_views": 85,
    "camera_poses": n_cam,
    "scene_centroid": centroid,
    "init_camera_position": init_pos,
    "psnr_test_mean": mean_test,
    "psnr_train_mean": mean_train,
    "psnr_test_per_image": [x["psnr"] for x in test],
}
with open(os.path.join(VIEWER, "metrics.json"), "w", encoding="utf-8") as fp:
    json.dump(metrics, fp, ensure_ascii=False, indent=2)

print("TEST  均值PSNR: %.2f dB  (n=%d)" % (mean_test, len(test)))
print("TRAIN 采样均值PSNR: %.2f dB  (n=%d)" % (mean_train, len(train)))
print("相机位姿数: %d  场景中心: %s  初始机位: %s" % (n_cam, centroid, init_pos))
print("已写出 gallery.json / metrics.json ->", VIEWER)
