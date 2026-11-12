"""坐标系变换：3DGS 世界系 -> 机器狗导航地图系。

这是"规划能落地"的第一道关卡
----------------------------
感知层（`perception/scene.py`）跑在 3DGS 重建坐标系里，约定写在源码注释里：
**X-Z 水平、Y 竖直**（LLFF/3DGS 前向采集约定），栅格 `occ[ix][iz]`。
机器狗厂家的导航栈是 ROS 约定：**X-Y 水平、Z 竖直**，`start_nav` 传
`{position:{x,y,z}, orientation:{x,y,z,w}}`。

两个坐标系不仅轴不同，**原点、朝向、尺度也几乎必然不同**（重建场景是任意原点的，
狗的地图是建图时的起始位姿）。所以这里不做"写死的公式"，而是给一个
**可标定的变换**：现场用几个同名地标标定后，填 `yaw_deg` / `translate` / `scale`。

默认约定（未标定时）
--------------------
- 轴映射：世界 (x, y, z) -> 地图 (X, Y, Z) = (x_r, -z_r, y)，即
  世界的 X 给地图 X，世界的竖直 Y 抬成地图的 Z，世界的 Z 取负给地图 Y。
  这是一组标准的手性一致的 Y-up -> Z-up 变换。
- 旋转：先在世界水平面绕竖直轴按 `yaw_deg` 旋转（俯视逆时针为正），再做轴映射。
- 平移 `translate` 在旋转之后的地图系上叠加；`scale` 同时作用于三轴。

标定点怎么填：拿两个在现场和地图里都能指认的点（如开关柜 A 角、门框），
解出 yaw 与 translate 填进配置即可，`fit_from_point_pairs()` 提供了最小二乘求解。
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field


class FrameTransform(BaseModel):
    """世界系（3DGS）到机器狗地图系的可标定刚体相似变换。

    world: X-Z 水平、Y 竖直（感知层坐标系）
    map:   X-Y 水平、Z 竖直（机器狗导航坐标系）
    """

    #: 绕世界竖直轴的旋转角（度，俯视逆时针为正）
    yaw_deg: float = 0.0
    #: 地图系平移（米），在旋转 + 轴映射之后叠加（第三分量仅在 use_world_height 时生效）
    translate: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    #: 统一尺度（1.0 = 等尺度；重建若已知有整体偏差可在此修正）
    scale: float = 1.0
    #: 地图系水平轴镜像（现场标定发现手性相反时启用，默认关闭）
    flip_x: bool = False
    flip_y: bool = False
    #: 导航目标点的地图 Z（地面导航恒为 0）
    nav_z: float = 0.0
    #: 是否把世界系高度 Y 带入地图 Z（平地为 False；多层/坡道场景才需要）
    use_world_height: bool = False
    name: str = "world3dgs->dogmap"

    # ------------------------------------------------------------------
    # 点变换
    # ------------------------------------------------------------------
    def world_to_map(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """世界坐标 (x,y,z) -> 机器狗地图坐标 (X,Y,Z)。"""
        th = math.radians(self.yaw_deg)
        c, s = math.cos(th), math.sin(th)
        xr = c * x - s * z
        zr = s * x + c * z
        X = xr * self.scale
        Y = -zr * self.scale
        Z = y * self.scale
        if self.flip_x:
            X = -X
        if self.flip_y:
            Y = -Y
        t = self.translate
        Z = (Z + t[2]) if (self.use_world_height and len(t) > 2) else self.nav_z
        return (round(X + t[0], 4), round(Y + t[1], 4), round(Z, 4))

    def map_to_world(self, X: float, Y: float, Z: float = 0.0) -> Tuple[float, float, float]:
        """逆变换：机器狗地图坐标 -> 世界坐标（用于把狗上报位姿画回 3DGS 场景）。"""
        t = self.translate
        x0 = (X - t[0]) / self.scale
        y0 = (Y - t[1]) / self.scale
        z0 = (Z - (t[2] if len(t) > 2 else 0.0)) / self.scale
        if self.flip_x:
            x0 = -x0
        if self.flip_y:
            y0 = -y0
        # 逆轴映射：X = x_r, Y = -z_r  =>  x_r = X, z_r = -Y
        xr, zr = x0, -y0
        th = math.radians(-self.yaw_deg)
        c, s = math.cos(th), math.sin(th)
        return (round(c * xr - s * zr, 4), round(z0, 4), round(s * xr + c * zr, 4))

    # ------------------------------------------------------------------
    # 朝向
    # ------------------------------------------------------------------
    def world_heading_to_map_yaw(self, dx: float, dz: float) -> float:
        """世界系水平方向向量 (dx, dz) -> 地图系 yaw（弧度，绕地图 Z 轴）。

        yaw=0 表示朝地图 +X 方向，符合 ROS 导航惯例。
        """
        xr, zr = self._rotate(dx, dz)
        return math.atan2(-zr, xr)

    def _rotate(self, x: float, z: float) -> Tuple[float, float]:
        th = math.radians(self.yaw_deg)
        c, s = math.cos(th), math.sin(th)
        return (c * x - s * z, s * x + c * z)

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Dict[str, float]:
        """yaw（弧度，绕 Z 轴）-> 四元数 {x,y,z,w}。"""
        return {"x": 0.0, "y": 0.0, "z": round(math.sin(yaw / 2.0), 6),
                "w": round(math.cos(yaw / 2.0), 6)}

    def orientation_towards(self, cur: Sequence[float], nxt: Sequence[float]) -> Dict[str, float]:
        """由当前点与下一点算出"朝向前进方向"的四元数（世界系入参）。"""
        dx = float(nxt[0]) - float(cur[0])
        dz = float(nxt[2]) - float(cur[2])
        if abs(dx) < 1e-9 and abs(dz) < 1e-9:
            return self.yaw_to_quaternion(0.0)
        return self.yaw_to_quaternion(self.world_heading_to_map_yaw(dx, dz))

    # ------------------------------------------------------------------
    # 标定
    # ------------------------------------------------------------------
    @classmethod
    def fit_from_point_pairs(cls, world_pts: Sequence[Sequence[float]],
                             map_pts: Sequence[Sequence[float]],
                             name: str = "calibrated") -> "FrameTransform":
        """用 >=2 组同名点（世界系 / 地图系）标定 yaw、scale、translate。

        只解水平面（导航只关心 X-Y），竖直分量按 `nav_z` 处理。
        采用相似变换的最小二乘闭式解（Umeyama 的 2D 简化版）。
        """
        if len(world_pts) < 2 or len(world_pts) != len(map_pts):
            raise ValueError("至少需要 2 组同名点，且两集合长度一致")
        n = len(world_pts)
        # 世界水平面取 (x, z)，并先按轴映射规则转成 (u, v) = (x, -z)
        W = [(float(p[0]), -float(p[2])) for p in world_pts]
        M = [(float(p[0]), float(p[1])) for p in map_pts]
        cw = (sum(u for u, _ in W) / n, sum(v for _, v in W) / n)
        cm = (sum(u for u, _ in M) / n, sum(v for _, v in M) / n)
        num_c = num_s = den = 0.0
        for (u, v), (U, V) in zip(W, M):
            du, dv = u - cw[0], v - cw[1]
            dU, dV = U - cm[0], V - cm[1]
            num_c += du * dU + dv * dV
            num_s += du * dV - dv * dU
            den += du * du + dv * dv
        if den <= 1e-12:
            raise ValueError("标定点退化（世界系点重合），无法求解")
        scale = math.hypot(num_c, num_s) / den
        # 注意方向：world_to_map 是"先绕世界竖直轴转 yaw，再取 Y=-z_r"，
        # 等价到 (u, v)=(x, -z) 平面上就是旋转 -yaw。故拟合出的 θ = -yaw。
        theta = math.atan2(num_s, num_c)
        yaw = -theta
        c, s = math.cos(theta), math.sin(theta)
        rx = scale * (c * cw[0] - s * cw[1])
        ry = scale * (s * cw[0] + c * cw[1])
        return cls(yaw_deg=math.degrees(yaw), scale=round(scale, 6),
                   translate=[round(cm[0] - rx, 4), round(cm[1] - ry, 4), 0.0],
                   name=name)

    def describe(self) -> str:
        return (f"{self.name}: yaw={self.yaw_deg:.2f}°, scale={self.scale:.4f}, "
                f"translate={[round(v, 3) for v in self.translate]}"
                f"{', flip_x' if self.flip_x else ''}{', flip_y' if self.flip_y else ''}")


def default_transform() -> FrameTransform:
    """未标定时的默认变换（纯轴映射，原点与朝向沿用重建坐标系）。

    **重要**：直接下真机前必须标定，否则航点会以重建原点为原点，
    与狗自身地图对不上。dry-run 导出与联调阶段可以用默认值。
    """
    return FrameTransform(name="default(axis-swap-only)")
