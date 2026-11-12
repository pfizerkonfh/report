import os, json, base64
BASE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(BASE)

def b64(path, mime):
    with open(path, 'rb') as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode('ascii')

# 指标
m = json.load(open(os.path.join(BASE, 'metrics.json'), encoding='utf-8'))

# 画廊
g = json.load(open(os.path.join(BASE, 'gallery.json'), encoding='utf-8'))
def resolve(p):
    return os.path.join(EXP, p.lstrip('/'))
gallery = {'test': [], 'train': []}
for setk in ['test', 'train']:
    for it in g[setk]:
        gallery[setk].append({
            'name': it['name'],
            'psnr': it['psnr'],
            'render': b64(resolve(it['render']), 'image/png'),
            'gt': b64(resolve(it['gt']), 'image/png'),
        })

# 旋转 GIF + 环绕动态 + 多角度静帧
rotation = b64(os.path.join(BASE, 'showcase_assets', 'rotation.gif'), 'image/gif')
orbit = b64(os.path.join(BASE, 'showcase_assets', 'scene_3dgs_vs_gt.gif'), 'image/gif')
views = []
view_labels = ['前视', '左视', '后视', '右视', '俯视', '仰视']
for i in range(6):
    views.append({'label': view_labels[i],
                  'img': b64(os.path.join(BASE, 'showcase_assets', f'view_{i}.png'), 'image/png')})

# 寻路演示（静态产物）
nav = {
    'topdown': b64(os.path.join(BASE, 'showcase_assets', 'nav_topdown.png'), 'image/png'),
    'path_gif': b64(os.path.join(BASE, 'showcase_assets', 'nav_path.gif'), 'image/gif'),
    'gif3d': b64(os.path.join(BASE, 'showcase_assets', 'nav_3d.gif'), 'image/gif'),
    'data': json.load(open(os.path.join(BASE, 'showcase_assets', 'nav_data.json'), encoding='utf-8')),
    'points': json.load(open(os.path.join(BASE, 'showcase_assets', 'nav_points.json'), encoding='utf-8')),
}

data = {
    'metrics': m,
    'gallery': gallery,
    'rotation': rotation,
    'orbit': orbit,
    'views': views,
    'nav': nav,
}
payload = json.dumps(data, ensure_ascii=False)

html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>实验室场景 · 3D 高斯泼溅建图可视化</title>
<style>
  :root{--bg:#f4f6fb;--card:#fff;--text:#1f2933;--muted:#647084;--accent:#2563eb;--accent2:#0ea5a4;--border:#e4e9f2;--good:#16a34a;--shadow:0 6px 24px rgba(30,50,90,.08)}
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text);line-height:1.6;-webkit-font-smoothing:antialiased}
  header.top{padding:30px 24px 22px;background:linear-gradient(120deg,#1e3a8a,#0ea5a4);color:#fff}
  header.top h1{margin:0 0 6px;font-size:24px;font-weight:700}
  header.top p{margin:0;opacity:.9;font-size:14px}
  .wrap{max-width:1180px;margin:0 auto;padding:0 20px 60px}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:-26px auto 26px;max-width:1140px}
  .stat{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px 18px;box-shadow:var(--shadow)}
  .stat .v{font-size:22px;font-weight:800;color:var(--accent)}
  .stat .l{font-size:12.5px;color:var(--muted);margin-top:2px}
  section{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:22px;margin:22px 0;box-shadow:var(--shadow)}
  section h2{margin:0 0 4px;font-size:18px}
  section .sub{margin:0 0 16px;color:var(--muted);font-size:13.5px}
  .toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:14px}
  .toggle{display:inline-flex;border:1px solid var(--border);border-radius:9px;overflow:hidden}
  .toggle button{border:none;border-radius:0;background:#fff;font:inherit;padding:8px 12px;cursor:pointer;color:var(--text)}
  .toggle button.on{background:var(--accent);color:#fff}
  .rotbox{width:100%;max-width:560px;margin:0 auto;border-radius:12px;overflow:hidden;background:#0b1220}
  .rotbox img{display:block;width:100%;height:auto}
  .rotcap{text-align:center;color:var(--muted);font-size:12.5px;margin-top:8px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px}
  .viewcard{border:1px solid var(--border);border-radius:12px;overflow:hidden;background:#0b1220}
  .viewcard img{display:block;width:100%;height:auto}
  .viewcard .cap{text-align:center;font-size:12.5px;color:var(--text);padding:6px 0;background:#fff}
  .grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:18px}
  .card{border:1px solid var(--border);border-radius:12px;overflow:hidden;background:#fff}
  .card .head{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;font-size:12.5px;color:var(--muted);border-bottom:1px solid var(--border)}
  .badge{background:#ecfdf5;color:var(--good);font-weight:700;padding:2px 8px;border-radius:20px;font-size:12px}
  .compare{position:relative;width:100%;aspect-ratio:288/141;background:#0b1220;user-select:none;touch-action:none}
  .compare img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block}
  .compare .gt{z-index:1}
  .compare .render{z-index:2;clip-path:inset(0 50% 0 0)}
  .compare .divider{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;z-index:3;transform:translateX(-1px);box-shadow:0 0 6px rgba(0,0,0,.5)}
  .compare .tag{position:absolute;top:8px;font-size:11px;color:#fff;background:rgba(0,0,0,.45);padding:2px 7px;border-radius:6px;z-index:4}
  .compare .tag.l{left:8px}.compare .tag.r{right:8px}
  .compare input[type=range]{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;z-index:5;margin:0}
  table.mt{width:100%;border-collapse:collapse;font-size:13.5px}
  table.mt th,table.mt td{border-bottom:1px solid var(--border);padding:9px 10px;text-align:left}
  table.mt th{color:var(--muted);font-weight:600;background:#fafbfe}
  .note{font-size:12.5px;color:var(--muted);margin-top:10px}
  .navwrap{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:18px;align-items:start}
  @media(max-width:820px){.navwrap{grid-template-columns:1fr}}
  .navcanvas{width:100%;border-radius:10px;background:#eef2f8;cursor:crosshair;display:block}
  .nav3d{width:100%;max-width:360px;border-radius:10px;background:#0b1220;display:block}
  .infobox{border:1px solid var(--border);border-radius:12px;padding:14px;background:#fafbfe}
  .infobox .row{display:flex;justify-content:space-between;font-size:13px;padding:5px 0;border-bottom:1px dashed var(--border)}
  .infobox .row:last-child{border:none}
  .infobox b{color:var(--accent)}
  .btn{border:1px solid var(--accent);background:#fff;color:var(--accent);border-radius:8px;padding:7px 12px;cursor:pointer;font:inherit}
  .btn:hover{background:var(--accent);color:#fff}
  .hint{font-size:12.5px;color:var(--muted);margin:8px 0 0}
  footer{text-align:center;color:var(--muted);font-size:12.5px;padding:20px}
  .legend-inline{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--muted);margin-top:8px}
  .legend-inline span{display:inline-flex;align-items:center;gap:6px}
  .sw{width:14px;height:14px;border-radius:3px;display:inline-block;border:1px solid rgba(0,0,0,.15)}
</style>
</head>
<body>
<header class="top">
  <h1>实验室场景 · 3D 高斯泼溅建图可视化</h1>
  <p>3D Gaussian Splatting 重建结果展示 — 旋转重建 / 多角度视图 / 渲染保真度量化 / 建图支撑寻路（含交互版）</p>
</header>
<div class="wrap">
  <div class="stats" id="stats"></div>

  <section>
    <h2>① 3D 建图结果（旋转动画）</h2>
    <p class="sub">由完整 3DGS 模型抽稀出的 12.9 万彩色高斯点云端到端渲染的旋转动画，直观体现重建出的三维实验室空间结构。</p>
    <div class="rotbox"><img id="rotgif" alt="旋转重建动画"></div>
    <div class="rotcap">自动循环旋转 · 体现几何结构与建图覆盖</div>
  </section>

  <section>
    <h2>② 多角度静态视图</h2>
    <p class="sub">同一重建点云从前、后、左、右、俯、仰六个视角观察，验证模型各方向的完整性。</p>
    <div class="grid" id="viewGrid"></div>
  </section>

  <section>
    <h2>③ 新视角合成质量（渲染 vs 真值）</h2>
    <p class="sub">拖动滑块对比 <b>3DGS 渲染图</b> 与 <b>真实照片(GT)</b>。左侧为模型在未见视角下的合成结果。</p>
    <div class="toolbar">
      <div class="toggle" id="splitToggle">
        <button data-set="test" class="on">测试集（13 视角）</button>
        <button data-set="train">训练集（采样 11 视角）</button>
      </div>
    </div>
    <div class="grid2" id="gallery"></div>
  </section>

  <section>
    <h2>④ 量化评估指标</h2>
    <p class="sub">基于测试集 / 训练集渲染与真值的逐像素 PSNR（峰值信噪比，单位 dB，越高越好）。</p>
    <table class="mt" id="metricsTable">
      <thead><tr><th>评估项</th><th>数值</th><th>说明</th></tr></thead>
      <tbody></tbody>
    </table>
    <p class="note" id="perImageNote"></p>
  </section>

  <section>
    <h2>⑤ 重建场景环绕动态（渲染 vs 真实）</h2>
    <p class="sub">将 85 张训练视角渲染图按相机绕场景中心的方位角排序，拼接成平滑环绕视频：<b>左为 3DGS 重建渲染、右为真实照片 GT</b>，直观对比重建保真度。</p>
    <div class="rotbox"><img id="orbitgif" alt="环绕动态对比"></div>
    <div class="rotcap">自动循环 · 左：3DGS 重建渲染 · 右：真实照片 GT</div>
  </section>

  <section>
    <h2>⑥ 建图如何支撑寻路（静态演示）</h2>
    <p class="sub">将 3DGS 点云投影为俯视 occupancy（可通行/障碍）地图，叠加相机真实走过的轨迹（蓝），并用 A* 规划出更短的避障捷径（橙）。</p>
    <div class="grid2">
      <div class="card"><div class="head"><span>俯视 occupancy 地图</span><span class="badge">A* 规划</span></div>
        <img src="__NAV_TOPDOWN__" alt="俯视地图" style="display:block;width:100%;height:auto;background:#eef2f8"></div>
      <div class="card"><div class="head"><span>智能体沿规划路径行进</span><span class="badge">动画</span></div>
        <img src="__NAV_PATHGIF__" alt="路径动画" style="display:block;width:100%;height:auto;background:#eef2f8"></div>
    </div>
    <div class="grid2" style="margin-top:18px">
      <div class="card"><div class="head"><span>规划路径贴回三维点云</span><span class="badge">旋转</span></div>
        <img src="__NAV_GIF3D__" alt="3D路径" style="display:block;width:100%;height:auto;background:#0b1220"></div>
      <div class="infobox" style="align-self:center">
        <div class="row"><span>场景尺度</span><b id="navScale"></b></div>
        <div class="row"><span>网格分辨率</span><b>0.20 m / 格</b></div>
        <div class="row"><span>自由 / 障碍 / 未知</span><b id="navCounts"></b></div>
        <div class="row"><span>默认规划路径长度</span><b id="navLen"></b></div>
        <p class="hint">说明：蓝线为采集时实际走过的探索轨迹，橙线为 A* 在地图上规划的更短避障路径——说明“有了这张重建地图，就能规划出比探索路径更优的路”。</p>
      </div>
    </div>
  </section>

  <section>
    <h2>⑦ 交互版：自由设定起点/终点实时规划</h2>
    <p class="sub">在左侧俯视地图上 <b>第 1 次点击设起点（绿）、第 2 次点击设终点（红）</b>，A* 会立即重新规划避障路径并联动右侧三维视图。纯 Canvas 实现，无需服务器 / WebGL。</p>
    <div class="navwrap">
      <div>
        <canvas id="navMap" class="navcanvas"></canvas>
        <div class="legend-inline">
          <span><i class="sw" style="background:#33415c"></i>障碍</span>
          <span><i class="sw" style="background:#eaf7ea"></i>可通行</span>
          <span><i class="sw" style="background:#c9d2e0"></i>未知（稀疏）</span>
          <span><i class="sw" style="background:#2563eb"></i>相机轨迹</span>
          <span><i class="sw" style="background:#f5821f"></i>规划路径</span>
          <span><i class="sw" style="background:#16a34a"></i>起点</span>
          <span><i class="sw" style="background:#dc2626"></i>终点</span>
        </div>
        <p class="hint" id="navHint">点击地图设置起点…</p>
      </div>
      <div>
        <canvas id="nav3d" class="nav3d" width="360" height="360"></canvas>
        <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
          <button class="btn" id="navReset">恢复默认起终点</button>
          <button class="btn" id="navSpin">暂停旋转</button>
        </div>
        <div class="infobox" style="margin-top:10px">
          <div class="row"><span>起点 (格)</span><b id="iStart">—</b></div>
          <div class="row"><span>终点 (格)</span><b id="iGoal">—</b></div>
          <div class="row"><span>路径长度</span><b id="iLen">—</b></div>
          <div class="row"><span>是否找到路径</span><b id="iFound">—</b></div>
        </div>
      </div>
    </div>
  </section>

  <footer>数据来源：exp/ 目录（3DGS 训练输出）· 本页为完全自包含单文件（所有图像与交互数据已内嵌，离线可用，无需服务器）</footer>
</div>

<script>
const DATA = __PAYLOAD__;
function el(id){return document.getElementById(id);}

// 统计卡
const mit = DATA.metrics;
const stats = [
  ['387,299','高斯数量 (iter 10000)'],
  [mit.iterations.toLocaleString(),'训练迭代次数'],
  [mit.camera_poses,'采集相机位姿'],
  [mit.psnr_test_mean+' dB','测试集 平均 PSNR'],
  [mit.psnr_train_mean+' dB','训练集 平均 PSNR'],
];
el('stats').innerHTML = stats.map(([v,l])=>`<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');

// 旋转 GIF / 环绕动态
el('rotgif').src = DATA.rotation;
el('orbitgif').src = DATA.orbit;

// 多角度
el('viewGrid').innerHTML = DATA.views.map(v=>`
  <div class="viewcard"><img src="${v.img}" alt="${v.label}"><div class="cap">${v.label}</div></div>`).join('');

// 画廊
function renderGallery(set){
  const data = DATA.gallery[set]||[];
  el('gallery').innerHTML = data.map(d=>`
    <div class="card">
      <div class="head"><span>${d.name}</span><span class="badge">PSNR ${d.psnr.toFixed(2)} dB</span></div>
      <div class="compare">
        <img class="gt" src="${d.gt}" alt="GT">
        <img class="render" src="${d.render}" alt="render">
        <div class="tag l">3DGS 渲染</div>
        <div class="tag r">真值 GT</div>
        <div class="divider"></div>
        <input type="range" min="0" max="100" value="50">
      </div>
    </div>`).join('');
  el('gallery').querySelectorAll('.compare').forEach(c=>{
    const slider=c.querySelector('input[type=range]'), render=c.querySelector('.render'), divider=c.querySelector('.divider');
    const apply=()=>{const v=+slider.value; render.style.clipPath=`inset(0 ${100-v}% 0 0)`; divider.style.left=v+'%';};
    slider.addEventListener('input', apply); apply();
  });
}
renderGallery('test');
el('splitToggle').querySelectorAll('button').forEach(b=>{
  b.onclick=()=>{ el('splitToggle').querySelectorAll('button').forEach(x=>x.classList.remove('on')); b.classList.add('on'); renderGallery(b.dataset.set); };
});

// 指标表
const rows = [
  ['场景类型', mit.scene, 'LLFF 前向采集的实验室场景'],
  ['模型格式', mit.model_format, 'Inria 3DGS .ply（含 SH 球谐）'],
  ['高斯数量', '387,299（iter10000）/ 375,387（iter7000）', '场景表征的基本图元数'],
  ['测试集 平均 PSNR', mit.psnr_test_mean+' dB', '13 个未参与训练的视角'],
  ['训练集 平均 PSNR', mit.psnr_train_mean+' dB', '85 个训练视角中采样 11 个'],
  ['逐视角 PSNR（测试）', mit.psnr_test_per_image.map(x=>x.toFixed(1)).join(' / '), '各测试图像峰值信噪比'],
];
el('metricsTable').querySelector('tbody').innerHTML = rows.map(r=>
  `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td style="color:var(--muted)">${r[2]}</td></tr>`).join('');
el('perImageNote').textContent =
  `说明：PSNR 越高代表合成图与真实照片越接近；3DGS 在训练视角可达约 ${mit.psnr_train_mean} dB，在未见视角约 ${mit.psnr_test_mean} dB，属该方法的典型水平。`;

// ===================== 寻路：静态信息 =====================
const NV = DATA.nav.data;
const RES = NV.res, XMIN = NV.xmin, ZMIN = NV.zmin, FLOORY = NV.floorY;
(function(){
  let free=0,occ=0,unk=0;
  NV.occ.forEach(col=>col.forEach(v=>{ if(v===1)occ++; else if(v===2)unk++; else free++; }));
  el('navScale').textContent = (NV.nx*RES).toFixed(1)+' m × '+(NV.nz*RES).toFixed(1)+' m';
  el('navCounts').textContent = free+' / '+occ+' / '+unk+' 格';
  el('navLen').textContent = pathMeters(NV.path).toFixed(2)+' m';
})();
function pathMeters(p){
  let L=0;
  for(let i=1;i<p.length;i++){ const dx=p[i][0]-p[i-1][0], dz=p[i][1]-p[i-1][1]; L+=Math.hypot(dx,dz)*RES; }
  return L;
}

// ===================== 交互版：2D 地图 =====================
const occ = NV.occ, NX = NV.nx, NZ = NV.nz;
const mapCanvas = el('navMap');
const cellPx = Math.max(7, Math.floor(760 / NX));
mapCanvas.width = NX*cellPx; mapCanvas.height = NZ*cellPx;
const mctx = mapCanvas.getContext('2d');
let curStart = NV.start.slice(), curGoal = NV.goal.slice(), curPath = NV.path.slice();
let pickState = 'start';

function drawDot(c, color, r){
  const px=c[0]*cellPx+cellPx/2, py=(NZ-1-c[1])*cellPx+cellPx/2;
  mctx.beginPath(); mctx.arc(px,py,r||Math.max(5,cellPx*0.8),0,7); mctx.fillStyle=color; mctx.fill();
  mctx.lineWidth=2; mctx.strokeStyle='#fff'; mctx.stroke();
}
function drawMap(){
  mctx.clearRect(0,0,mapCanvas.width,mapCanvas.height);
  for(let x=0;x<NX;x++) for(let z=0;z<NZ;z++){
    const v=occ[x][z];
    mctx.fillStyle = v===1 ? '#33415c' : v===2 ? '#c9d2e0' : '#eaf7ea';
    mctx.fillRect(x*cellPx,(NZ-1-z)*cellPx,cellPx,cellPx);
  }
  // 轨迹
  mctx.strokeStyle='#2563eb'; mctx.lineWidth=2; mctx.beginPath();
  NV.traj.forEach(([x,z],i)=>{ const px=x*cellPx+cellPx/2, py=(NZ-1-z)*cellPx+cellPx/2; i?mctx.lineTo(px,py):mctx.moveTo(px,py); });
  mctx.stroke();
  // 规划路径
  if(curPath && curPath.length){
    mctx.strokeStyle='#f5821f'; mctx.lineWidth=3; mctx.beginPath();
    curPath.forEach(([x,z],i)=>{ const px=x*cellPx+cellPx/2, py=(NZ-1-z)*cellPx+cellPx/2; i?mctx.lineTo(px,py):mctx.moveTo(px,py); });
    mctx.stroke();
  }
  if(curStart) drawDot(curStart,'#16a34a');
  if(curGoal) drawDot(curGoal,'#dc2626');
}
drawMap();

function astar(s, g){
  const key=(x,z)=>x+','+z;
  const G={}; G[key(s[0],s[1])]=0; const came={};
  const open=[[0,0,s[0],s[1]]];
  const h=(x,z)=>Math.hypot(x-g[0],z-g[1]);
  while(open.length){
    let bi=0; for(let i=1;i<open.length;i++) if(open[i][0]<open[bi][0]) bi=i;
    const best=open.splice(bi,1)[0]; const f=best[0], x=best[2], z=best[3];
    if(x===g[0]&&z===g[1]){ const path=[[x,z]]; let c=key(x,z); while(came[c]!==undefined){ c=came[c]; const p=c.split(','); path.push([+p[0],+p[1]]); } return path.reverse(); }
    for(let dx=-1;dx<=1;dx++) for(let dz=-1;dz<=1;dz++){
      if(!dx&&!dz) continue; const nx=x+dx, nz=z+dz;
      if(nx<0||nz<0||nx>=NX||nz>=NZ) continue;
      if(occ[nx][nz]===1) continue;
      if(dx&&dz && (occ[x+dx][z]===1 || occ[x][z+dz]===1)) continue; // 不穿墙角
      const ng=G[key(x,z)] + ((dx&&dz)?1.41421356:1);
      if(ng < (G[key(nx,nz)] ?? 1e9)){ G[key(nx,nz)]=ng; came[key(nx,nz)]=key(x,z); open.push([ng+h(nx,nz), h(nx,nz), nx, nz]); }
    }
  }
  return null;
}

function updateInfo(){
  el('iStart').textContent = curStart.join(', ');
  el('iGoal').textContent = curGoal.join(', ');
  el('iLen').textContent = curPath ? pathMeters(curPath).toFixed(2)+' m' : '—';
  el('iFound').textContent = curPath ? ('是（'+curPath.length+' 格）') : '否（被障碍隔离）';
  // 联动 3D
  threeDraw();
}
mapCanvas.addEventListener('click', (e)=>{
  const r=mapCanvas.getBoundingClientRect();
  const mx=(e.clientX-r.left)*(mapCanvas.width/r.width);
  const my=(e.clientY-r.top)*(mapCanvas.height/r.height);
  const x=Math.floor(mx/cellPx), z=NZ-1-Math.floor(my/cellPx);
  if(x<0||z<0||x>=NX||z>=NZ) return;
  if(pickState==='start'){ curStart=[x,z]; pickState='goal'; el('navHint').textContent='已设起点（绿），请点击设置终点（红）…'; }
  else { curGoal=[x,z]; pickState='start';
    curPath = astar(curStart, curGoal);
    el('navHint').textContent='已重新规划，可继续点击设置新起点/终点。';
  }
  drawMap(); updateInfo();
});
el('navReset').onclick=()=>{ curStart=NV.start.slice(); curGoal=NV.goal.slice(); curPath=NV.path.slice(); pickState='start'; drawMap(); updateInfo(); el('navHint').textContent='已恢复默认起终点。'; };

// ===================== 交互版：3D 回贴（纯 Canvas 旋转） =====================
const PT = DATA.nav.points;
const PX = PT.pos, PC = PT.col, NP = PX.length;
let cc=[0,0,0];
for(let i=0;i<NP;i++){ cc[0]+=PX[i][0]; cc[1]+=PX[i][1]; cc[2]+=PX[i][2]; }
cc=cc.map(v=>v/NP);
for(let i=0;i<NP;i++){ PX[i][0]-=cc[0]; PX[i][1]-=cc[1]; PX[i][2]-=cc[2]; }
let lim=0; for(let i=0;i<NP;i++){ lim=Math.max(lim, Math.abs(PX[i][0]), Math.abs(PX[i][1]), Math.abs(PX[i][2])); } lim*=1.1;
const c3 = el('nav3d'); const ctx3 = c3.getContext('2d');
let yaw=0, spinning=true;
function threeDraw(){
  const W=c3.width, H=c3.height; ctx3.fillStyle='#0b1220'; ctx3.fillRect(0,0,W,H);
  const ca=Math.cos(yaw), sa=Math.sin(yaw); const s=Math.min(W,H)/(2*lim);
  // 点
  for(let i=0;i<NP;i++){
    const x=PX[i][0], y=PX[i][1], z=PX[i][2];
    const rx=x*ca - z*sa, rz=x*sa + z*ca;
    const sx=W/2 + rx*s, sy=H/2 - y*s;
    if(sx<-5||sx>W+5||sy<-5||sy>H+5) continue;
    ctx3.fillStyle='rgb('+PC[i][0]+','+PC[i][1]+','+PC[i][2]+')';
    ctx3.fillRect(sx,sy,1.4,1.4);
  }
  // 路径
  if(curPath && curPath.length){
    ctx3.strokeStyle='#f5821f'; ctx3.lineWidth=3; ctx3.beginPath();
    curPath.forEach(([gx,gz],i)=>{
      const wx=XMIN+(gx+0.5)*RES - cc[0], wy=FLOORY+0.3 - cc[1], wz=ZMIN+(gz+0.5)*RES - cc[2];
      const rx=wx*ca - wz*sa, rz=wx*sa + wz*ca;
      const sx=W/2+rx*s, sy=H/2-wy*s; i?ctx3.lineTo(sx,sy):ctx3.moveTo(sx,sy);
    });
    ctx3.stroke();
    // 终点标记
    const lg=curPath[curPath.length-1];
    const wx=XMIN+(lg[0]+0.5)*RES-cc[0], wy=FLOORY+0.3-cc[1], wz=ZMIN+(lg[1]+0.5)*RES-cc[2];
    const rx=wx*ca - wz*sa, rz=wx*sa + wz*ca; const sx=W/2+rx*s, sy=H/2-wy*s;
    ctx3.beginPath(); ctx3.arc(sx,sy,5,0,7); ctx3.fillStyle='#dc2626'; ctx3.fill();
  }
}
function loop(){ if(spinning) yaw+=0.012; threeDraw(); requestAnimationFrame(loop); }
el('navSpin').onclick=()=>{ spinning=!spinning; el('navSpin').textContent = spinning?'暂停旋转':'继续旋转'; };
loop();
updateInfo();
</script>
</body>
</html>
"""

html = html.replace('__PAYLOAD__', payload)
html = html.replace('__NAV_TOPDOWN__', data['nav']['topdown'])
html = html.replace('__NAV_PATHGIF__', data['nav']['path_gif'])
html = html.replace('__NAV_GIF3D__', data['nav']['gif3d'])
out = os.path.join(BASE, 'showcase.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print('written', out, 'size(MB)=', round(os.path.getsize(out)/1e6, 2))
