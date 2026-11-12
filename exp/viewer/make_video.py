import os, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

BASE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(BASE)
OUT = os.path.join(BASE, 'showcase_assets')
os.makedirs(OUT, exist_ok=True)

cams = json.load(open(os.path.join(EXP, 'cameras.json')))
# 训练集取前 85 个相机（与 train/ours_10000/renders 00000..00084 对应）
train_ids = list(range(85))
pos = np.array([cams[i]['position'] for i in train_ids], dtype=float)
cx, cy, cz = pos[:,0].mean(), pos[:,1].mean(), pos[:,2].mean()
# 按绕场景中心的方位角排序，形成平滑环绕
angles = [math.atan2(pos[i,2]-cz, pos[i,0]-cx) for i in train_ids]
order_az = sorted(range(85), key=lambda k: angles[k])
order_seq = list(range(85))   # 原始采集时序 00000 -> 00084

def font(size):
    try:
        return ImageFont.truetype("msyh.ttc", size)
    except Exception:
        return ImageFont.load_default()

def make_frames(scale, order):
    ph = int(141*scale)            # 单面板高度
    pw = int(288*scale)            # 单面板宽度
    gap = 8
    label_h = max(28, int(22*scale))
    W = pw*2 + gap
    H = label_h + ph
    title_f = font(int(16*scale))
    small_f = font(int(12*scale))
    bg = (11,18,32)
    frames = []
    for rank, k in enumerate(order):
        i = train_ids[k]
        rp = os.path.join(EXP, f'train/ours_10000/renders/{i:05d}.png')
        gp = os.path.join(EXP, f'train/ours_10000/gt/{i:05d}.png')
        r = Image.open(rp).convert('RGB').resize((pw, ph), Image.LANCZOS)
        g = Image.open(gp).convert('RGB').resize((pw, ph), Image.LANCZOS) if os.path.exists(gp) else Image.new('RGB',(pw,ph),(40,40,40))
        frame = Image.new('RGB', (W, H), bg)
        frame.paste(r, (0, label_h))
        frame.paste(g, (pw+gap, label_h))
        d = ImageDraw.Draw(frame)
        d.text((W//2, label_h//2), "实验室场景 · 3D 高斯泼溅重建 vs 真实照片",
                font=title_f, fill=(255,255,255), anchor='mm')
        d.text((pw//2, label_h+10), "3DGS 渲染", font=small_f, fill=(200,220,255), anchor='mm')
        d.text((pw+gap+pw//2, label_h+10), "真实照片 GT", font=small_f, fill=(180,255,235), anchor='mm')
        frames.append(np.array(frame))
    return frames

def export(order, tag, title):
    name = f'scene_3dgs_vs_gt_{tag}' if tag else 'scene_3dgs_vs_gt'
    print(f'rendering {name} (2x)...', flush=True)
    frames_hi = make_frames(2.0, order)
    mp4 = os.path.join(OUT, name+'.mp4')
    writer = imageio.get_writer(mp4, fps=10, codec='libx264', quality=8,
                                macro_block_size=None, output_params=['-pix_fmt','yuv420p'])
    for f in frames_hi:
        writer.append_data(f)
    writer.close()
    print(f'mp4 bytes:', os.path.getsize(mp4), flush=True)
    print(f'rendering {name} gif (1x)...', flush=True)
    frames_lo = make_frames(1.0, order)
    gif = os.path.join(OUT, name+'.gif')
    imageio.mimsave(gif, frames_lo, fps=10, loop=0)
    print(f'gif bytes:', os.path.getsize(gif), flush=True)

# 方位角环绕（保留原名，供 showcase.html 内嵌）
export(order_az, '', '环绕视角')
# 原始采集时序
export(order_seq, 'sequence', '采集时序')

print('DONE', flush=True)
