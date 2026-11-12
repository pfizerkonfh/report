import os, re, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(BASE)
OUT = os.path.join(BASE, 'showcase_assets')
os.makedirs(OUT, exist_ok=True)

# ---------- 读取点云（与相机同坐标系） ----------
with open(os.path.join(BASE, 'points_light.ply'), 'rb') as f:
    raw = f.read()
end = raw.find(b'end_header\n') + len(b'end_header\n')
header = raw[:end].decode('latin-1'); data = raw[end:]
n = int(re.search(r'element vertex (\d+)', header).group(1))
arr = np.frombuffer(data[:n*15], dtype=np.dtype([('xyz','<f4',(3,)),('rgb','u1',(3,))]), count=n)
xyz = arr['xyz'].astype(np.float32)

# 裁剪离群高斯（保留 0.5%~99.5% 区间），避免极值污染
lo, hi = np.percentile(xyz, 0.5, axis=0), np.percentile(xyz, 99.5, axis=0)
mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
P = xyz[mask]
print('裁剪后点数:', len(P), '/', n, flush=True)

X, Y, Zc = P[:,0], P[:,1], P[:,2]

# ---------- 构建俯视 occupancy 网格 ----------
res = 0.20  # 米/格
xmin, xmax = X.min(), X.max()
zmin, zmax = Zc.min(), Zc.max()
nx = int(np.ceil((xmax-xmin)/res)) + 1
nz = int(np.ceil((zmax-zmin)/res)) + 1
print('grid', nx, 'x', nz, flush=True)

# ---------- 相机轨迹（记录的真实路径，需先于 occupancy 计算） ----------
cj = json.load(open(os.path.join(EXP, 'cameras.json')))
pos = np.array([c['position'] for c in cj])  # 98 个相机
traj = pos[:85]  # 训练集 85 视角（采集时序）
tx = traj[:,0]; tz = traj[:,2]
tgx = np.clip(np.floor((tx-xmin)/res).astype(int), 0, nx-1)
tgz = np.clip(np.floor((tz-zmin)/res).astype(int), 0, nz-1)
traj_cells = list(zip(tgx, tgz))

gx = np.floor((X-xmin)/res).astype(int)
gz = np.floor((Zc-zmin)/res).astype(int)
# 每格统计：点数 / 高度范围(稳健 p95-p5)
cnt = np.zeros((nx, nz), dtype=np.int32)
ymin = np.full((nx, nz), np.inf)
ymax = np.full((nx, nz), -np.inf)
order = np.lexsort((gz, gx))
ugx, ugz = gx[order], gz[order]
uy = Y[order]
# 累积点数
np.add.at(cnt, (ugx, ugz), 1)
# y 范围：用分位数近似，逐格计算
cell_y = {}
for i in range(len(ugx)):
    cell_y.setdefault((ugx[i], ugz[i]), []).append(uy[i])
# 每格：地面高度(10%分位) 与 高于地面 0.6m 的点数
height = np.zeros((nx, nz), dtype=np.float32)
above_cnt = np.zeros((nx, nz), dtype=np.int32)
for (ix, iz), ys in cell_y.items():
    ys = np.array(ys)
    floorY = np.percentile(ys, 10)
    above_cnt[ix, iz] = int((ys > floorY + 0.6).sum())

# occupancy: 0 自由, 1 障碍, 2 未知
OCC = np.zeros((nx, nz), dtype=np.int8)
OCC[above_cnt >= 3] = 1
# 已知自由：相机实际经过的格子及其邻域 -> 强制为自由（机器人走过即可行）
traj_free = np.zeros((nx, nz), dtype=bool)
for (cx, cz) in set(traj_cells):
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            nx3, nz3 = cx+dx, cz+dz
            if 0 <= nx3 < nx and 0 <= nz3 < nz:
                traj_free[nx3, nz3] = True
OCC[traj_free] = 0
OCC[(cnt < 4) & (OCC == 0)] = 2  # 点太少 -> 未知
# 障碍膨胀 1 格（机器人半径余量），但保留已知自由走廊
from scipy.ndimage import binary_dilation
occ2 = binary_dilation(OCC == 1, iterations=1)
OCC[occ2] = 1
OCC[traj_free] = 0   # 强制恢复已知自由路径

def nearest_free(cx, cz):
    if OCC[cx, cz] != 1:
        return (cx, cz)
    for r in range(1, 20):
        for dx in range(-r, r+1):
            for dz in range(-r, r+1):
                nx2, nz2 = cx+dx, cz+dz
                if 0 <= nx2 < nx and 0 <= nz2 < nz and OCC[nx2, nz2] == 0:
                    return (nx2, nz2)
    return (cx, cz)

start = nearest_free(*traj_cells[0])
# 目标：取轨迹上离起点最远（且非同格）的点，使路径真正横跨场景、需绕障
best, bestd = traj_cells[0], -1
for c in traj_cells:
    d = math.hypot(c[0]-start[0], c[1]-start[1])
    if d > bestd:
        bestd, best = d, c
goal = nearest_free(*best)
print('start', start, 'goal', goal, 'maxdist', round(bestd,1), flush=True)

# ---------- A* 寻路（8 连通） ----------
def astar(start, goal):
    import heapq
    sx, sz = start; gx2, gz2 = goal
    open_ = [(0, 0, sx, sz)]
    came = {}
    g = {(sx, sz): 0}
    while open_:
        f, _, x, z = heapq.heappop(open_)
        if (x, z) == (gx2, gz2):
            path = [(x, z)]; c = (x, z)
            while c in came:
                c = came[c]; path.append(c)
            return path[::-1]
        for dx in (-1,0,1):
            for dz in (-1,0,1):
                if dx==0 and dz==0: continue
                nx3, nz3 = x+dx, z+dz
                if not (0 <= nx3 < nx and 0 <= nz3 < nz): continue
                if OCC[nx3, nz3] == 1: continue
                if (dx and dz) and (OCC[x+dx, z]==1 or OCC[x, z+dz]==1): continue  # 不穿墙角
                ng = g[(x,z)] + (1.414 if dx and dz else 1.0)
                if ng < g.get((nx3,nz3), 1e9):
                    g[(nx3,nz3)] = ng
                    came[(nx3,nz3)] = (x, z)
                    h = math.hypot(nx3-gx2, nz3-gz2)
                    heapq.heappush(open_, (ng+h, h, nx3, nz3))
    return None

def flood_fill(s):
    seen=set([s]); st=[s]
    while st:
        x,z=st.pop()
        for dx in (-1,0,1):
            for dz in (-1,0,1):
                if dx==0 and dz==0: continue
                nx3,nz3=x+dx,z+dz
                if 0<=nx3<nx and 0<=nz3<nz and OCC[nx3,nz3]!=1 and (nx3,nz3) not in seen:
                    seen.add((nx3,nz3)); st.append((nx3,nz3))
    return seen

path = astar(start, goal)
if path is None:
    reach = flood_fill(start)
    cand = [c for c in reach if OCC[c[0],c[1]]==0]
    cand.sort(key=lambda c: math.hypot(c[0]-goal[0], c[1]-goal[1]))
    if cand:
        goal = cand[0]
        path = astar(start, goal)
print('A* path len:', len(path) if path else None, 'goal', goal, flush=True)

# ---------- 坐标转换：格 -> 像素 ----------
px_per = 11
W, Hh = nx*px_per, nz*px_per
def c2p(cx, cz):  # 注意 Z 轴朝上 -> 图像 y 翻转
    return (int(cx*px_per + px_per/2), int(Hh - (cz*px_per + px_per/2)))

# ---------- 绘制静态俯视地图 ----------
def font(sz):
    try: return ImageFont.truetype("msyh.ttc", sz)
    except: return ImageFont.load_default()
fbig, fsm = font(20), font(13)

img = Image.new('RGB', (W, Hh), (238,242,247))
d = ImageDraw.Draw(img)
for ix in range(nx):
    for iz in range(nz):
        if OCC[ix,iz]==1: col=(51,65,92)
        elif OCC[ix,iz]==2: col=(201,210,224)
        else: col=(234,247,234)
        x0=ix*px_per; y0=Hh-(iz+1)*px_per
        d.rectangle([x0,y0,x0+px_per-1,y0+px_per-1], fill=col)
# 记录轨迹（蓝，半透明感用实线）
for i in range(len(traj_cells)-1):
    a=c2p(*traj_cells[i]); b=c2p(*traj_cells[i+1])
    d.line([a,b], fill=(37,99,235), width=3)
# A* 规划路径（橙，粗）
if path:
    for i in range(len(path)-1):
        a=c2p(*path[i]); b=c2p(*path[i+1])
        d.line([a,b], fill=(245,130,32), width=4)
# 起点/终点
s=c2p(*start); g=c2p(*goal)
d.ellipse([s[0]-7,s[1]-7,s[0]+7,s[1]+7], fill=(22,163,74), outline=(255,255,255))
d.ellipse([g[0]-7,g[1]-7,g[0]+7,g[1]+7], fill=(220,38,38), outline=(255,255,255))
# 图例（左上，避免右边缘溢出）
d.rectangle([8,8, 8+244, 8+140], fill=(255,255,255), outline=(200,208,220))
d.text((18,14), "俯视 occupancy 地图", font=fbig, fill=(30,40,60))
swy=48
for col,label in [((51,65,92),"障碍"),((234,247,234),"可通行"),((201,210,224),"未知")]:
    d.rectangle([18,swy,34,swy+12], fill=col, outline=(120,130,150))
    d.text((42,swy+6), label, font=fsm, fill=(60,70,90), anchor='lm')
    swy+=18
d.line([18,swy+4, 40,swy+4], fill=(37,99,235), width=3)
d.text((46,swy+4), "相机记录轨迹（探索路径）", font=fsm, fill=(60,70,90), anchor='lm')
swy+=18
d.line([18,swy+4, 40,swy+4], fill=(245,130,32), width=4)
d.text((46,swy+4), "A* 规划路径（更短捷径）", font=fsm, fill=(60,70,90), anchor='lm')
# 比例尺 (1m)
bar_len = int(1.0/res*px_per)
d.line([(14, Hh-22),(14+bar_len, Hh-22)], fill=(30,40,60), width=3)
d.text((14, Hh-18), "1 m", font=fsm, fill=(30,40,60))
img.save(os.path.join(OUT, 'nav_topdown.png'))
print('saved nav_topdown.png', img.size, flush=True)

# ---------- 动画：智能体沿规划路径移动 ----------
frames = []
path_pts = [c2p(*c) for c in (path or [])]
nframes = max(len(path_pts), 2)
for i in range(nframes):
    f = img.copy()
    dd = ImageDraw.Draw(f)
    # 已走过路径高亮
    for j in range(min(i, len(path_pts)-1)):
        dd.line([path_pts[j], path_pts[j+1]], fill=(16,185,129), width=4)
    if path_pts:
        cur = path_pts[min(i, len(path_pts)-1)]
        dd.ellipse([cur[0]-9,cur[1]-9,cur[0]+9,cur[1]+9], fill=(250,204,21), outline=(120,53,15))
    dd.text((W//2, 12), f"智能体沿地图规划路径行进  {i+1}/{nframes}", font=fsm, fill=(30,40,60), anchor='mm')
    frames.append(f)
gifp = os.path.join(OUT, 'nav_path.gif')
frames[0].save(gifp, save_all=True, append_images=frames[1:], duration=80, loop=0)
print('saved nav_path.gif frames', len(frames), flush=True)

# ---------- 3D：路径贴回点云旋转展示 ----------
print('rendering 3D path gif...', flush=True)
sub = np.random.default_rng(1).choice(len(P), size=min(16000, len(P)), replace=False)
sp = P[sub]; cr = np.clip(arr['rgb'][mask][sub]/255.0*1.7 + 0.06, 0, 1)
# 路径 3D 坐标（格中心 + 地面高度）
floorY = np.percentile(Y, 3)
path3 = []
for (cx,cz) in (path or []):
    wx = xmin + (cx+0.5)*res
    wz = zmin + (cz+0.5)*res
    path3.append((wx, floorY+0.3, wz))
path3 = np.array(path3)
# 居中
cc = sp.mean(0); sp -= cc
if len(path3): path3 = path3 - cc
lim = float(np.percentile(np.abs(sp), 99))*1.15
def rot(yaw):
    a=np.radians(yaw); ca,sa=np.cos(a),np.sin(a)
    return sp[:,0]*ca - sp[:,2]*sa, sp[:,1], sp[:,2]*sa + sp[:,0]*ca
f3=[]
for i in range(36):
    x2,y2,z2 = rot(i*10)
    fig=plt.figure(figsize=(5,5), dpi=110); ax=fig.add_axes([0,0,1,1], projection='3d')
    ax.set_facecolor('#0b1220'); fig.patch.set_facecolor('#0b1220')
    ax.scatter(x2,z2,y2,c=cr,s=1.6,depthshade=False,linewidths=0)
    if len(path3):
        px2 = path3[:,0]*np.cos(np.radians(i*10)) - path3[:,2]*np.sin(np.radians(i*10))
        pz2 = path3[:,0]*np.sin(np.radians(i*10)) + path3[:,2]*np.cos(np.radians(i*10))
        ax.plot(px2, pz2, path3[:,1], color='#f5821f', linewidth=2.5)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.set_box_aspect([1,1,1]); ax.view_init(elev=22, azim=0); ax.axis('off')
    fig.savefig(os.path.join(OUT,'_f3.png'), facecolor=fig.get_facecolor(), dpi=110); plt.close(fig)
    f3.append(Image.open(os.path.join(OUT,'_f3.png')).convert('RGB'))
g3 = os.path.join(OUT, 'nav_3d.gif')
f3[0].save(g3, save_all=True, append_images=f3[1:], duration=90, loop=0)
tp = os.path.join(OUT, '_f3.png')
if os.path.exists(tp): os.remove(tp)
print('saved nav_3d.gif', flush=True)

# ---------- 导出交互数据（供 showcase.html 交互版使用） ----------
nav_data = {
    'nx': nx, 'nz': nz, 'res': res,
    'xmin': float(xmin), 'zmin': float(zmin),
    'floorY': float(floorY),
    'occ': OCC.astype(int).tolist(),          # 0 自由 / 1 障碍 / 2 未知
    'traj': [[int(c[0]), int(c[1])] for c in traj_cells],
    'start': [int(start[0]), int(start[1])],
    'goal':  [int(goal[0]),  int(goal[1])],
    'path':  [[int(c[0]), int(c[1])] for c in (path or [])],
}
json.dump(nav_data, open(os.path.join(OUT, 'nav_data.json'), 'w'), ensure_ascii=False)

# 三维交互点云（降采样，供交互版 3D 回贴）
rng = np.random.default_rng(7)
keep = rng.choice(len(P), size=min(6000, len(P)), replace=False)
nav_points = {
    'pos': np.round(P[keep].astype(np.float32), 3).tolist(),
    'col': arr['rgb'][mask][keep].astype(int).tolist(),
}
json.dump(nav_points, open(os.path.join(OUT, 'nav_points.json'), 'w'), ensure_ascii=False)
print('saved nav_data.json + nav_points.json', flush=True)
print('DONE', flush=True)
