import os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="批量统计results.json指标")
parser.add_argument("--root_dir", type=str, required=True, help="所有场景的根目录，例如 ./output/llff9")
parser.add_argument("--output_file", type=str, default="./metrics_summary.txt", help="汇总结果保存路径")
args = parser.parse_args()

ROOT_DIR = args.root_dir
SUMMARY_FILE = args.output_file

all_metrics = []
scene_names = []

for scene_name in os.listdir(ROOT_DIR):
    print(scene_name)
    scene_path = os.path.join(ROOT_DIR, scene_name)
    if not os.path.isdir(scene_path):
        continue

    json_path = os.path.join(scene_path, "results.json")
    if not os.path.exists(json_path):
        print(f"跳过 {scene_name}：未找到 results.json")
        continue

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data["ours_10000"]
        psnr = metrics["PSNR"]
        ssim = metrics["SSIM"]
        lpips = metrics["LPIPS"]

        scene_names.append(scene_name)
        all_metrics.append([psnr, ssim, lpips])
        print(f"已读取：{scene_name} | PSNR: {psnr} | SSIM: {ssim} | LPIPS: {lpips}")

    except Exception as e:
        print(f"读取 {scene_name} 失败：{str(e)}")

all_metrics = np.array(all_metrics)
psnr_list = all_metrics[:, 0]
ssim_list = all_metrics[:, 1]
lpips_list = all_metrics[:, 2]

avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
avg_lpips = np.mean(lpips_list)

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write(f"{'scene':<10} {'PSNR':<12} {'SSIM':<12} {'LPIPS':<12}\n")
    f.write("-" * 60 + "\n")

    for i, scene in enumerate(scene_names):
        f.write(f"{scene:<10} {psnr_list[i]:<12} {ssim_list[i]:<12} {lpips_list[i]:<12}\n")

    # mean
    f.write("-" * 60 + "\n")
    f.write(f"{'mean':<10} {avg_psnr:<12} {avg_ssim:<12} {avg_lpips:<12}\n")

print("\n" + "=" * 60)
print("汇总文件保存至：", SUMMARY_FILE)
print("全部场景平均值：")
print(f"平均 PSNR: {avg_psnr}")
print(f"平均 SSIM: {avg_ssim}")
print(f"平均 LPIPS: {avg_lpips}")
print("=" * 60)
