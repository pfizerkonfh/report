"""感知层：把 3DGS 实验室场景重建产物接入决策系统。

背景
----
此前系统是「纯逻辑仿真」：camera 工具返回一个 uuid 假路径，占据/可达性全靠布尔 flag 手填。
本模块把真实的 3DGS 重建结果（俯视 occupancy 栅格 + 点云 + 重建质量指标）接进来，
使规划与前置校验第一次建立在**真实场景数据**之上。

数据来源（3DGS 训练输出，由 `exp/viewer/build_nav.py` 生成）
----------------------------------------------------------
- `nav_data.json`：俯视 occupancy 栅格（0 自由 / 1 障碍 / 2 未知）+ 相机轨迹 + A* 参考路径
- `nav_points.json`：降采样三维点云（位置 + 颜色），用于高度/密度探测
- `metrics.json`：重建质量指标（PSNR、高斯数量、相机位姿数）

坐标系约定（与 build_nav.py 严格一致，切勿自行改动）
--------------------------------------------------
- **X-Z 为水平面，Y 为竖直轴**（3DGS/LLFF 前向采集约定）。
- 栅格索引 `occ[ix][iz]`，`ix` 对应 X，`iz` 对应 Z。
- 格 -> 世界：`wx = xmin + (ix+0.5)*res`，`wz = zmin + (iz+0.5)*res`。
- 世界 -> 格：`ix = floor((wx-xmin)/res)`，`iz = floor((wz-zmin)/res)`。
- 地面高度 `floorY` 由点云 Y 的 3% 分位估计。
"""
from __future__ import annotations

import json
import math
import os
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 占据栅格取值（与 build_nav.py 保持一致）
FREE = 0
OCCUPIED = 1
UNKNOWN = 2

_CELL_NAMES = {FREE: "自由", OCCUPIED: "障碍", UNKNOWN: "未知"}


def _read_json(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _locate(base: str, filename: str) -> str:
    """在 base / base/showcase_assets / base/.. 三处查找文件，返回存在的路径或空串。"""
    if not base:
        return ""
    for candidate in (
        os.path.join(base, filename),
        os.path.join(base, "showcase_assets", filename),
        os.path.join(os.path.dirname(base), filename),
    ):
        if os.path.exists(candidate):
            return candidate
    return ""


class SceneModel:
    """真实场景模型：占据栅格 + 点云 + 重建质量。

    所有查询（可通行性、连通性、路径规划、高度探测）都是对**真实重建数据**的计算，
    不再是硬编码布尔量。
    """

    def __init__(
        self,
        occ: Sequence[Sequence[int]],
        nx: int,
        nz: int,
        res: float,
        xmin: float,
        zmin: float,
        floor_y: float,
        traj: Optional[Sequence[Sequence[int]]] = None,
        start: Optional[Sequence[int]] = None,
        goal: Optional[Sequence[int]] = None,
        reference_path: Optional[Sequence[Sequence[int]]] = None,
        points: Optional[Sequence[Sequence[float]]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        source: str = "",
    ) -> None:
        self.occ = [list(row) for row in occ]
        self.nx = int(nx)
        self.nz = int(nz)
        self.res = float(res)
        self.xmin = float(xmin)
        self.zmin = float(zmin)
        self.floor_y = float(floor_y)
        self.traj = [list(c) for c in (traj or [])]
        self.start = list(start) if start else None
        self.goal = list(goal) if goal else None
        self.reference_path = [list(c) for c in (reference_path or [])]
        self._points = points
        self.metrics = metrics or {}
        self.source = source

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    @classmethod
    def from_dir(cls, base: str) -> Optional["SceneModel"]:
        """从 3DGS 产物目录加载。base 可以是 showcase_assets 或其父目录 viewer。"""
        nav_path = _locate(base, "nav_data.json")
        if not nav_path:
            return None
        nav = _read_json(nav_path) or {}
        if "occ" not in nav:
            return None

        pts_path = _locate(base, "nav_points.json")
        pts = None
        if pts_path:
            pdata = _read_json(pts_path) or {}
            if isinstance(pdata.get("pos"), list):
                pts = pdata["pos"]

        metrics = _read_json(_locate(base, "metrics.json")) or {}

        return cls(
            occ=nav["occ"],
            nx=nav.get("nx", len(nav["occ"])),
            nz=nav.get("nz", len(nav["occ"][0]) if nav["occ"] else 0),
            res=nav.get("res", 0.2),
            xmin=nav.get("xmin", 0.0),
            zmin=nav.get("zmin", 0.0),
            floor_y=nav.get("floorY", 0.0),
            traj=nav.get("traj"),
            start=nav.get("start"),
            goal=nav.get("goal"),
            reference_path=nav.get("path"),
            points=pts,
            metrics=metrics,
            source=nav_path,
        )

    @classmethod
    def discover(cls, start_dir: Optional[str] = None, levels: int = 6) -> Optional["SceneModel"]:
        """从 start_dir 逐级向上寻找 3DGS 产物目录，找到即加载。"""
        cur = os.path.abspath(start_dir or os.path.dirname(os.path.abspath(__file__)))
        for _ in range(levels):
            for sub in ("", "exp/viewer/showcase_assets", "exp/viewer", "exp"):
                cand = os.path.join(cur, sub) if sub else cur
                model = cls.from_dir(cand)
                if model is not None:
                    return model
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return None

    # ------------------------------------------------------------------
    # 坐标转换
    # ------------------------------------------------------------------
    def world_to_cell(self, x: float, z: float) -> Tuple[int, int]:
        ix = int(math.floor((float(x) - self.xmin) / self.res))
        iz = int(math.floor((float(z) - self.zmin) / self.res))
        return ix, iz

    def cell_to_world(self, ix: int, iz: int) -> Tuple[float, float]:
        return (self.xmin + (ix + 0.5) * self.res,
                self.zmin + (iz + 0.5) * self.res)

    def cell_to_world3(self, ix: int, iz: int, height: float = 0.3) -> Tuple[float, float, float]:
        """格 -> 三维世界坐标（高度默认贴地 0.3m，模拟机器人机身中心）。"""
        wx, wz = self.cell_to_world(ix, iz)
        return (wx, self.floor_y + height, wz)

    def in_bounds(self, ix: int, iz: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iz < self.nz

    # ------------------------------------------------------------------
    # 占据查询
    # ------------------------------------------------------------------
    def cell_value(self, ix: int, iz: int) -> int:
        if not self.in_bounds(ix, iz):
            return UNKNOWN
        return int(self.occ[ix][iz])

    def is_free(self, ix: int, iz: int) -> bool:
        return self.cell_value(ix, iz) == FREE

    def is_occupied(self, ix: int, iz: int) -> bool:
        return self.cell_value(ix, iz) == OCCUPIED

    def is_unknown(self, ix: int, iz: int) -> bool:
        return self.cell_value(ix, iz) == UNKNOWN

    def traversable(self, ix: int, iz: int, allow_unknown: bool = True) -> bool:
        v = self.cell_value(ix, iz)
        if v == OCCUPIED:
            return False
        if v == UNKNOWN:
            return allow_unknown
        return True

    def nearest_free(self, ix: int, iz: int, allow_unknown: bool = True,
                     max_radius: int = 25) -> Optional[Tuple[int, int]]:
        """若该格不可通行，螺旋搜索最近的可通行格（真机目标点纠偏必需）。"""
        ix = max(0, min(self.nx - 1, int(ix)))
        iz = max(0, min(self.nz - 1, int(iz)))
        if self.traversable(ix, iz, allow_unknown):
            return (ix, iz)
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if max(abs(dx), abs(dz)) != r:
                        continue
                    cx, cz = ix + dx, iz + dz
                    if self.in_bounds(cx, cz) and self.traversable(cx, cz, allow_unknown):
                        return (cx, cz)
        return None

    # ------------------------------------------------------------------
    # 连通性
    # ------------------------------------------------------------------
    def reachable_from(self, ix: int, iz: int, allow_unknown: bool = True) -> set:
        """8 连通洪泛，返回可达格集合（用于判断目标是否真的能走到）。"""
        start = (int(ix), int(iz))
        if not self.in_bounds(*start) or not self.traversable(*start, allow_unknown):
            return set()
        seen = {start}
        stack = [start]
        while stack:
            x, z = stack.pop()
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dz == 0:
                        continue
                    cx, cz = x + dx, z + dz
                    if (cx, cz) in seen or not self.in_bounds(cx, cz):
                        continue
                    if not self.traversable(cx, cz, allow_unknown):
                        continue
                    seen.add((cx, cz))
                    stack.append((cx, cz))
        return seen

    def connected(self, a: Tuple[int, int], b: Tuple[int, int],
                  allow_unknown: bool = True) -> bool:
        return b in self.reachable_from(a[0], a[1], allow_unknown)

    # ------------------------------------------------------------------
    # A* 路径规划（真实计算）
    # ------------------------------------------------------------------
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int],
             allow_unknown: bool = True) -> Optional[List[Tuple[int, int]]]:
        """8 连通 A*，禁止穿墙角。返回格序列（含起终点），不可达返回 None。"""
        sx, sz = int(start[0]), int(start[1])
        gx, gz = int(goal[0]), int(goal[1])
        if not (self.in_bounds(sx, sz) and self.in_bounds(gx, gz)):
            return None
        if not (self.traversable(sx, sz, allow_unknown) and self.traversable(gx, gz, allow_unknown)):
            return None
        if (sx, sz) == (gx, gz):
            return [(sx, sz)]

        open_: List[Tuple[float, float, int, int]] = [(0.0, 0.0, sx, sz)]
        came: Dict[Tuple[int, int], Tuple[int, int]] = {}
        gscore: Dict[Tuple[int, int], float] = {(sx, sz): 0.0}

        while open_:
            _, _, x, z = heappop(open_)
            if (x, z) == (gx, gz):
                path = [(x, z)]
                cur = (x, z)
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dz == 0:
                        continue
                    cx, cz = x + dx, z + dz
                    if not self.in_bounds(cx, cz):
                        continue
                    if not self.traversable(cx, cz, allow_unknown):
                        continue
                    # 不穿墙角：斜向移动要求两个正交邻格都通行
                    if dx and dz:
                        if not (self.traversable(x + dx, z, allow_unknown)
                                and self.traversable(x, z + dz, allow_unknown)):
                            continue
                    step = math.sqrt(2.0) if (dx and dz) else 1.0
                    ng = gscore[(x, z)] + step
                    if ng < gscore.get((cx, cz), float("inf")):
                        gscore[(cx, cz)] = ng
                        came[(cx, cz)] = (x, z)
                        h = math.hypot(cx - gx, cz - gz)
                        heappush(open_, (ng + h, h, cx, cz))
        return None

    def plan_world(self, start_xyz: Sequence[float], goal_xyz: Sequence[float],
                   allow_unknown: bool = True, snap: bool = True
                   ) -> Dict[str, Any]:
        """在世界坐标下规划。返回格路径、世界路径、真实里程（米）与未知区占比。

        snap=True 时把落在障碍上的起终点纠偏到最近可通行格（真机常见情况）。
        """
        s_cell = self.world_to_cell(start_xyz[0], start_xyz[2])
        g_cell = self.world_to_cell(goal_xyz[0], goal_xyz[2])
        snapped = False
        if snap:
            fs = self.nearest_free(s_cell[0], s_cell[1], allow_unknown)
            fg = self.nearest_free(g_cell[0], g_cell[1], allow_unknown)
            if fs and fs != s_cell:
                s_cell, snapped = fs, True
            if fg and fg != g_cell:
                g_cell, snapped = fg, True

        path = self.plan(s_cell, g_cell, allow_unknown)
        if not path:
            return {
                "success": False,
                "reason": "目标不可达（障碍阻断或位于未知区）",
                "start_cell": list(s_cell), "goal_cell": list(g_cell),
                "snapped": snapped,
            }

        world = [self.cell_to_world3(c[0], c[1]) for c in path]
        length_m = sum(
            math.hypot(world[i + 1][0] - world[i][0], world[i + 1][2] - world[i][2])
            for i in range(len(world) - 1)
        )
        unknown_cells = sum(1 for c in path if self.is_unknown(c[0], c[1]))
        return {
            "success": True,
            "cells": [list(c) for c in path],
            "world_path": [[round(v, 3) for v in p] for p in world],
            "length_m": round(length_m, 2),
            "steps": len(path),
            "unknown_cells": unknown_cells,
            "unknown_ratio": round(unknown_cells / max(len(path), 1), 3),
            "start_cell": list(s_cell), "goal_cell": list(g_cell),
            "snapped": snapped,
        }

    # ------------------------------------------------------------------
    # 点云探测（高度 / 密度）—— 无人机需净空，机器狗需地面可通行
    # ------------------------------------------------------------------
    def probe(self, x: float, z: float, radius: float = 0.5) -> Dict[str, Any]:
        """以 (x,z) 为中心、radius 为半径，统计真实点云的高度分布与密度。"""
        if not self._points:
            return {"available": False, "reason": "未加载点云数据"}
        r2 = radius * radius
        ys: List[float] = []
        n = 0
        for p in self._points:
            dx = p[0] - x
            dz = p[2] - z
            if dx * dx + dz * dz <= r2:
                ys.append(p[1])
                n += 1
        if n == 0:
            return {"available": True, "points": 0, "density": 0.0,
                    "reason": "该位置附近无重建点（可能是空洞/未观测区）"}
        ys.sort()
        lo, hi = ys[0], ys[-1]
        mid = ys[n // 2]
        return {
            "available": True,
            "points": n,
            "radius_m": radius,
            "y_min": round(lo, 3), "y_max": round(hi, 3), "y_median": round(mid, 3),
            "clearance_above_floor_m": round(hi - self.floor_y, 3),
            "vertical_extent_m": round(hi - lo, 3),
        }

    # ------------------------------------------------------------------
    # 统计与摘要
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        free = occ = unk = 0
        for ix in range(self.nx):
            for iz in range(self.nz):
                v = self.cell_value(ix, iz)
                if v == FREE:
                    free += 1
                elif v == OCCUPIED:
                    occ += 1
                else:
                    unk += 1
        total = max(self.nx * self.nz, 1)
        return {
            "grid": [self.nx, self.nz],
            "resolution_m": self.res,
            "extent_m": [round(self.nx * self.res, 2), round(self.nz * self.res, 2)],
            "floor_y": round(self.floor_y, 3),
            "cells_total": total,
            "cells_free": free, "cells_occupied": occ, "cells_unknown": unk,
            "free_ratio": round(free / total, 3),
            "known_ratio": round((free + occ) / total, 3),
            "explored_cells": len(set(tuple(c) for c in self.traj)),
            "reference_path_steps": len(self.reference_path),
            "point_cloud_size": len(self._points) if self._points else 0,
        }

    def quality(self) -> Dict[str, Any]:
        m = self.metrics or {}
        return {
            "scene": m.get("scene", ""),
            "model_format": m.get("model_format", ""),
            "iterations": m.get("iterations"),
            "gaussian_count": m.get("gaussian_count_10000") or m.get("gaussian_count_7000"),
            "camera_poses": m.get("camera_poses"),
            "psnr_test_mean": m.get("psnr_test_mean"),
            "psnr_train_mean": m.get("psnr_train_mean"),
        }

    def default_goal(self) -> Tuple[int, int]:
        if self.goal:
            return (self.goal[0], self.goal[1])
        return (self.nx // 2, self.nz // 2)

    def default_start(self) -> Tuple[int, int]:
        if self.start:
            return (self.start[0], self.start[1])
        if self.traj:
            return (self.traj[0][0], self.traj[0][1])
        return (0, 0)

    def to_dict(self, include_grid: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "source": self.source,
            "stats": self.stats(),
            "quality": self.quality(),
            "start": list(self.default_start()),
            "goal": list(self.default_goal()),
        }
        if include_grid:
            out.update({
                "grid": [self.nx, self.nz],
                "res": self.res,
                "origin": [self.xmin, self.zmin],
                "occ": self.occ,
                "traj": self.traj,
                "path": self.reference_path,
            })
        return out

    def to_facts(self) -> Dict[str, Any]:
        """把真实场景统计导出为 WorldState facts，供 Precondition 语义校验使用。

        这一步是「感知层真正接入决策」的关键：前置条件不再是手填布尔量，
        而是可以判定 `scene_known_ratio >= 0.5`、`scene_loaded == True` 这类真实事实。
        """
        s, q = self.stats(), self.quality()
        return {
            "scene_loaded": True,
            "scene_source": self.source,
            "scene_grid": f"{s['grid'][0]}x{s['grid'][1]}",
            "scene_extent_m": s["extent_m"],
            "scene_resolution_m": s["resolution_m"],
            "scene_known_ratio": s["known_ratio"],
            "scene_free_cells": s["cells_free"],
            "scene_occupied_cells": s["cells_occupied"],
            "scene_unknown_cells": s["cells_unknown"],
            "scene_psnr_test": q.get("psnr_test_mean"),
            "scene_gaussians": q.get("gaussian_count"),
            "scene_camera_poses": q.get("camera_poses"),
        }

    def describe(self) -> str:
        s, q = self.stats(), self.quality()
        return (
            f"3DGS 重建场景：栅格 {s['grid'][0]}x{s['grid'][1]}@{s['resolution_m']}m "
            f"（{s['extent_m'][0]}x{s['extent_m'][1]}m），"
            f"可通行 {s['cells_free']} 格 / 障碍 {s['cells_occupied']} 格 / 未知 {s['cells_unknown']} 格，"
            f"已知率 {s['known_ratio']:.0%}；重建 PSNR(test) {q.get('psnr_test_mean')}dB，"
            f"高斯数 {q.get('gaussian_count')}，相机位姿 {q.get('camera_poses')}。"
        )

    def _trim_points(self, max_points: int) -> None:
        if self._points and max_points > 0 and len(self._points) > max_points:
            self._points = list(self._points)[:max_points]


def load_scene(data_dir: str = "", enabled: bool = True,
               max_points: int = 6000) -> Optional["SceneModel"]:
    """按配置加载真实 3DGS 场景；未找到或已禁用则返回 None（系统优雅降级）。"""
    if not enabled:
        return None
    model: Optional[SceneModel] = None
    if data_dir:
        model = SceneModel.from_dir(data_dir)
    if model is None:
        model = SceneModel.discover()
    if model is not None:
        model._trim_points(max_points)
    return model
