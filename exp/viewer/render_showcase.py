import os, json, struct, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'showcase_assets')
os.makedirs(OUT, exist_ok=True)

PLY = os.path.join(BASE, 'points_light.ply')

# ---- 读取二进制 PLY (x,y,z f4 + r,g,b u1) ----
with open(PLY, 'rb') as f:
    raw = f.read()
end = raw.find(b'end_header\n') + len(b'end_header\n')
header = raw[:end].decode('latin-1')
data = raw[end:]
import re
n = int(re.search(r'element vertex (\d+)', header).group(1))
dtype = np.dtype([('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])
arr = np.frombuffer(data[:n*15], dtype=dtype, count=n)
xyz = arr['xyz'].astype(np.float32)
rgb = arr['rgb'].astype(np.float32) / 255.0

# 抽稀用于动画（加速 + 控体积）
rng = np.random.default_rng(0)
sub = rng.choice(n, size=min(16000, n), replace=False)
px, py, pz = xyz[sub,0], xyz[sub,1], xyz[sub,2]
cr, cg, cb = rgb[sub,0], rgb[sub,1], rgb[sub,2]

# 居中
cx, cy, cz = px.mean(), py.mean(), pz.mean()
px -= cx; py -= cy; pz -= cz

# 统一坐标范围，避免每帧比例跳变
lim = float(np.percentile(np.abs(np.stack([px,py,pz])), 99)) * 1.15

def rotate(yaw_deg):
    a = np.radians(yaw_deg)
    ca, sa = np.cos(a), np.sin(a)
    x2 = px*ca - pz*sa
    z2 = px*sa + pz*ca
    return x2, py, z2

print('rendering frames...', flush=True)
W = H = 460
frames = []
NFR = 36
for i in range(NFR):
    yaw = i * (360.0 / NFR)
    x2, y2, z2 = rotate(yaw)
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = fig.add_axes([0,0,1,1], projection='3d')
    ax.set_facecolor('#0b1220')
    fig.patch.set_facecolor('#0b1220')
    ax.scatter(x2, z2, y2, c=np.stack([cr,cg,cb],1), s=1.1, depthshade=False, linewidths=0)
    # 注意：matplotlib 3D 轴顺序是 (X, Y, Z)；这里把 y(高度) 放到第三个轴
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=18, azim=0)
    ax.axis('off')
    p = os.path.join(OUT, f'frame_{i:03d}.png')
    fig.savefig(p, facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    frames.append(p)
    if i % 6 == 0:
        print('  frame', i, flush=True)

# 合成 GIF
print('assembling gif...', flush=True)
imgs = [Image.open(p).convert('RGB') for p in frames]
gif_path = os.path.join(OUT, 'rotation.gif')
imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=90, loop=0,
             optimize=False, disposal=2)
for p in frames:
    os.remove(p)
print('gif bytes:', os.path.getsize(gif_path), flush=True)

# ---- 多角度静帧 ----
angles = [(20,'前视'), (110,'左视'), (200,'后视'), (290,'右视'), (20,'俯视-高'), (20,'仰视-低')]
elevs  = [18, 18, 18, 18, 62, -30]
for idx,(az,label) in enumerate(angles):
    x2, y2, z2 = rotate(az)
    fig = plt.figure(figsize=(3.4,3.4), dpi=110)
    ax = fig.add_axes([0,0,1,1], projection='3d')
    ax.set_facecolor('#0b1220'); fig.patch.set_facecolor('#0b1220')
    ax.scatter(x2, z2, y2, c=np.stack([cr,cg,cb],1), s=1.3, depthshade=False, linewidths=0)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=elevs[idx], azim=0)
    ax.axis('off')
    out = os.path.join(OUT, f'view_{idx}.png')
    fig.savefig(out, facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    print('  view', idx, label, flush=True)

print('DONE', flush=True)
