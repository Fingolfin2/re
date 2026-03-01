#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 GetTopology.ipynb 与 AddressMatch.ipynb 重构为一个独立脚本。

功能概览：
1. 基于道路 shp 构建有向路网拓扑（节点、边、长度、道路等级、预计通行时间）。
2. 导出路网节点坐标（npy）与边信息（csv）。
3. 将卡口点位与路网节点做最近邻匹配（带阈值和一对一约束），并导出结果。

使用方式示例：
1) 先在“显式配置区”修改输入输出路径；
2) 运行：python topology_address_pipeline.py --mode all
   或：python topology_address_pipeline.py --mode topology
   或：python topology_address_pipeline.py --mode match
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import LineString, MultiLineString, Point

# ============================ 显式配置区（请按需修改） ============================
# 注意：
# 1. road_shp_path：道路路网 shp 文件（字段需包含 osm_id / oneway / fclass / maxspeed）。
# 2. bayonet_shp_path：卡口点位 shp 文件（默认使用 KKBH 作为卡口编号字段）。
# 3. output_dir：所有输出文件保存目录（若不存在会自动创建）。
CONFIG = {
    "road_shp_path": r"D:\your_data\中心城区路网地图OSM.shp",
    "bayonet_shp_path": r"D:\your_data\中心城区卡口点位.shp",
    "output_dir": r".\output",
    "road_graph_filename": "center_roadnet.gml",
    "node_position_filename": "center_pos.npy",
    "edges_csv_filename": "center_edges.csv",
    "match_npy_filename": "center_realpos.npy",
    "match_csv_filename": "center_realpos.csv",
    "bayonet_id_field": "KKBH",
    "match_threshold_m": 60.0,
    # 下面参数沿用 notebook 中的经验值：
    "avg_speed_kmh": 35.73,  # 无 maxspeed 时的默认路网平均速度
    "motorway_speed_kmh": 100.0,  # 高速路默认速度
    "trunk_speed_kmh": 80.0,  # 快速路默认速度
    "maxspeed_discount": 0.9,  # 对 maxspeed 乘折减系数，贴近货车实际运行
    "lat_degree_per_meter": 8.993216e-06,  # 1 米对应纬度变化（近似值）
    "lon_degree_per_meter": 9.769874e-06,  # 1 米对应经度变化（23°纬度附近）
    "progress_step": 200,  # 每处理多少条道路打印一次进度
    "verbose": True,
}
# =============================================================================


@dataclass
class AppConfig:
    """运行配置：由显式配置区 + CLI 覆盖项共同决定。"""

    road_shp_path: Path
    bayonet_shp_path: Path
    output_dir: Path
    road_graph_filename: str
    node_position_filename: str
    edges_csv_filename: str
    match_npy_filename: str
    match_csv_filename: str
    bayonet_id_field: str
    match_threshold_m: float
    avg_speed_kmh: float
    motorway_speed_kmh: float
    trunk_speed_kmh: float
    maxspeed_discount: float
    lat_degree_per_meter: float
    lon_degree_per_meter: float
    progress_step: int
    verbose: bool

    @property
    def road_graph_path(self) -> Path:
        return self.output_dir / self.road_graph_filename

    @property
    def node_position_path(self) -> Path:
        return self.output_dir / self.node_position_filename

    @property
    def edges_csv_path(self) -> Path:
        return self.output_dir / self.edges_csv_filename

    @property
    def match_npy_path(self) -> Path:
        return self.output_dir / self.match_npy_filename

    @property
    def match_csv_path(self) -> Path:
        return self.output_dir / self.match_csv_filename


# OSM fclass -> 道路等级编号（与 notebook 保持一致）
FCLASS_TO_KIND = {
    "motorway": 1,
    "motorway_link": 1,
    "trunk": 2,
    "trunk_link": 2,
    "primary": 3,
    "primary_link": 3,
    "secondary": 4,
    "secondary_link": 4,
    "tertiary": 5,
    "tertiary_link": 5,
    "unclassified": 6,
    "residential": 6,
    "living_street": 7,
}

# 道路等级编号 -> 中文名称（用于边信息导出）
KIND_TO_NAME = {
    1: "高速公路",
    2: "城市快速路",
    3: "主干道",
    4: "次干道",
    5: "支路",
    6: "一般道路",
    7: "生活街区道路",
    0: "未知",
}


def to_path(path_str: str, base_dir: Path) -> Path:
    """将配置中的路径字符串转换为 Path，并把相对路径解析到脚本目录。"""
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(args: argparse.Namespace) -> AppConfig:
    """加载配置：先用 CONFIG，再应用命令行覆盖。"""
    script_dir = Path(__file__).resolve().parent

    road_shp_path = args.road_shp or str(CONFIG["road_shp_path"])
    bayonet_shp_path = args.bayonet_shp or str(CONFIG["bayonet_shp_path"])
    output_dir = args.output_dir or str(CONFIG["output_dir"])
    match_threshold = (
        args.match_threshold
        if args.match_threshold is not None
        else float(CONFIG["match_threshold_m"])
    )

    return AppConfig(
        road_shp_path=to_path(road_shp_path, script_dir),
        bayonet_shp_path=to_path(bayonet_shp_path, script_dir),
        output_dir=to_path(output_dir, script_dir),
        road_graph_filename=str(CONFIG["road_graph_filename"]),
        node_position_filename=str(CONFIG["node_position_filename"]),
        edges_csv_filename=str(CONFIG["edges_csv_filename"]),
        match_npy_filename=str(CONFIG["match_npy_filename"]),
        match_csv_filename=str(CONFIG["match_csv_filename"]),
        bayonet_id_field=str(CONFIG["bayonet_id_field"]),
        match_threshold_m=float(match_threshold),
        avg_speed_kmh=float(CONFIG["avg_speed_kmh"]),
        motorway_speed_kmh=float(CONFIG["motorway_speed_kmh"]),
        trunk_speed_kmh=float(CONFIG["trunk_speed_kmh"]),
        maxspeed_discount=float(CONFIG["maxspeed_discount"]),
        lat_degree_per_meter=float(CONFIG["lat_degree_per_meter"]),
        lon_degree_per_meter=float(CONFIG["lon_degree_per_meter"]),
        progress_step=int(CONFIG["progress_step"]),
        verbose=bool(CONFIG["verbose"]),
    )


def validate_input_file(path: Path, label: str) -> None:
    """检查输入文件是否存在，避免后续报错不直观。"""
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在：{path}")


def parse_maxspeed(value: object) -> float:
    """
    解析 maxspeed 字段为 km/h 数值。

    常见情况：
    - 数字：60 / 80.0
    - 字符串："60" / "60 km/h" / "60;80"
    - 空值或异常：返回 0（后续走默认速度逻辑）
    """
    if value is None or pd.isna(value):
        return 0.0
    if isinstance(value, (int, float, np.number)):
        return max(0.0, float(value))

    text = str(value).strip().lower()
    if not text:
        return 0.0
    # 遇到 "60;80" 这类值时，取第一个速度
    text = re.split(r"[;,/|]", text)[0]
    match = re.search(r"\d+(\.\d+)?", text)
    if not match:
        return 0.0
    return max(0.0, float(match.group(0)))


def normalize_oneway(oneway: object) -> str:
    """标准化 oneway：F 正向、T 反向、B 双向（默认）。"""
    value = str(oneway).strip().upper() if oneway is not None else "B"
    return value if value in {"F", "T", "B"} else "B"


def unique_coords_in_order(coords: Iterable[Sequence[float]]) -> List[Tuple[float, float]]:
    """按原顺序去重坐标，避免重复点造成零长度边。"""
    seen = set()
    result: List[Tuple[float, float]] = []
    for coord in coords:
        point = (float(coord[0]), float(coord[1]))
        if point in seen:
            continue
        seen.add(point)
        result.append(point)
    return result


def extract_points(geometry: object) -> List[Tuple[float, float]]:
    """
    从道路几何中提取点序列。

    - LineString：直接读取坐标。
    - MultiLineString：严格复现 notebook 口径，固定取 geoms[1]（第二段）。
    """
    if geometry is None:
        return []
    if isinstance(geometry, LineString):
        return unique_coords_in_order(geometry.coords)
    if isinstance(geometry, MultiLineString):
        geoms = list(geometry.geoms)
        if len(geoms) < 2:
            # 严格复现 notebook 的 geoms[1] 口径：当分段不足两段时，直接报错。
            # 这样可以避免静默改口径导致结果偏移。
            raise ValueError(
                f"MultiLineString 分段数量不足 2（实际 {len(geoms)}），无法按 notebook 的 geoms[1] 规则处理。"
            )
        return unique_coords_in_order(geoms[1].coords)
    return []


def geometry_type_name(geom: object) -> str:
    """返回几何对象类型名，便于日志输出。"""
    if geom is None:
        return "None"
    gtype = getattr(geom, "geom_type", None)
    return str(gtype) if gtype else type(geom).__name__


def estimate_duration_seconds(
    length_m: float,
    maxspeed_kmh: float,
    fclass: str,
    config: AppConfig,
) -> float:
    """根据道路属性估算通行时间（秒），逻辑继承 notebook。"""
    if maxspeed_kmh > 0:
        speed_kmh = config.maxspeed_discount * maxspeed_kmh
    elif fclass in {"motorway", "motorway_link"}:
        speed_kmh = config.motorway_speed_kmh
    elif fclass in {"trunk", "trunk_link"}:
        speed_kmh = config.trunk_speed_kmh
    else:
        speed_kmh = config.avg_speed_kmh

    # 防守式处理，避免极端值导致除零。
    speed_kmh = max(speed_kmh, 1e-6)
    return length_m / (speed_kmh / 3.6)


def promote_to_link_name(old_name: str, road_id: str) -> str:
    """
    将节点名提升为 LINK 命名并追加道路 ID。

    例：
    - O-39092939 + 4001 -> LINK-39092939-4001
    - LINK-39092939-4001 + 5002 -> LINK-39092939-4001-5002
    """
    merged_name = f"{old_name}-{road_id}"
    parts = merged_name.split("-", 1)
    return f"LINK-{parts[1]}" if len(parts) == 2 else f"LINK-{merged_name}"


def resolve_or_create_node(
    graph: nx.DiGraph,
    coord_to_node: Dict[Tuple[float, float], str],
    point: Tuple[float, float],
    road_id: str,
    default_name: str,
) -> str:
    """
    根据坐标获取或创建节点。

    如果该坐标已出现，说明遇到道路交汇点：
    - 将旧节点名升级为 LINK-*；
    - 在图中原地重命名（copy=False）以减小开销；
    - 更新坐标索引。
    """
    old_name = coord_to_node.get(point)
    if old_name is None:
        graph.add_node(default_name)
        coord_to_node[point] = default_name
        return default_name

    new_name = promote_to_link_name(old_name, road_id)
    if new_name != old_name:
        nx.relabel_nodes(graph, {old_name: new_name}, copy=False)
    coord_to_node[point] = new_name
    return new_name


def add_edge(
    graph: nx.DiGraph,
    start_node: str,
    end_node: str,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    oneway: str,
    fclass: str,
    maxspeed_kmh: float,
    config: AppConfig,
) -> None:
    """按 oneway 方向向图中添加边，并写入 length/kind/duration 属性。"""
    length_m = geodesic(
        (start_point[1], start_point[0]),
        (end_point[1], end_point[0]),
    ).m
    kind = FCLASS_TO_KIND.get(fclass, 0)
    duration_s = estimate_duration_seconds(length_m, maxspeed_kmh, fclass, config)

    if oneway == "F":
        graph.add_edge(start_node, end_node, length=length_m, kind=kind, duration=duration_s)
    elif oneway == "T":
        graph.add_edge(end_node, start_node, length=length_m, kind=kind, duration=duration_s)
    else:
        graph.add_edge(start_node, end_node, length=length_m, kind=kind, duration=duration_s)
        graph.add_edge(end_node, start_node, length=length_m, kind=kind, duration=duration_s)


def build_topology(config: AppConfig) -> Dict[str, Tuple[float, float]]:
    """构建路网拓扑并输出图与节点坐标。"""
    validate_input_file(config.road_shp_path, "路网shp")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    road_gdf = gpd.read_file(config.road_shp_path)
    graph = nx.DiGraph()
    coord_to_node: Dict[Tuple[float, float], str] = {}

    start_time = time.time()
    total_rows = len(road_gdf)
    print(f"[拓扑] 开始处理道路记录：{total_rows}")

    for idx, road in road_gdf.iterrows():
        road_id = str(road.get("osm_id", idx))
        oneway = normalize_oneway(road.get("oneway", "B"))
        fclass = str(road.get("fclass", "") or "")
        maxspeed_kmh = parse_maxspeed(road.get("maxspeed", 0))
        points = extract_points(road.geometry)

        if len(points) < 2:
            continue

        last_node_name: Optional[str] = None
        last_point: Optional[Tuple[float, float]] = None

        for point_idx, point in enumerate(points):
            if point_idx == 0:
                node_name = resolve_or_create_node(
                    graph=graph,
                    coord_to_node=coord_to_node,
                    point=point,
                    road_id=road_id,
                    default_name=f"O-{road_id}",
                )
            elif point_idx == len(points) - 1:
                node_name = resolve_or_create_node(
                    graph=graph,
                    coord_to_node=coord_to_node,
                    point=point,
                    road_id=road_id,
                    default_name=f"D-{road_id}",
                )
                if last_node_name is None or last_point is None:
                    last_node_name = node_name
                    last_point = point
                    continue
                add_edge(
                    graph=graph,
                    start_node=last_node_name,
                    end_node=node_name,
                    start_point=last_point,
                    end_point=point,
                    oneway=oneway,
                    fclass=fclass,
                    maxspeed_kmh=maxspeed_kmh,
                    config=config,
                )
            else:
                node_name = resolve_or_create_node(
                    graph=graph,
                    coord_to_node=coord_to_node,
                    point=point,
                    road_id=road_id,
                    default_name=f"{point_idx}-{road_id}",
                )
                if last_node_name is None or last_point is None:
                    last_node_name = node_name
                    last_point = point
                    continue
                add_edge(
                    graph=graph,
                    start_node=last_node_name,
                    end_node=node_name,
                    start_point=last_point,
                    end_point=point,
                    oneway=oneway,
                    fclass=fclass,
                    maxspeed_kmh=maxspeed_kmh,
                    config=config,
                )

            last_node_name = node_name
            last_point = point

        if config.verbose and (idx + 1) % max(1, config.progress_step) == 0:
            print(f"[拓扑] 进度：{idx + 1}/{total_rows}")

    # coord_to_node 是“坐标 -> 节点名”，这里转换为“节点名 -> 坐标”以便和原 notebook 保持一致。
    pos = {node_name: coord for coord, node_name in coord_to_node.items()}

    nx.write_gml(graph, config.road_graph_path)
    np.save(config.node_position_path, pos)

    total_length_m = sum(attr.get("length", 0.0) for _, _, attr in graph.edges(data=True))
    elapsed_s = time.time() - start_time
    print(f"[拓扑] 节点数：{graph.number_of_nodes()}")
    print(f"[拓扑] 边数：{graph.number_of_edges()}")
    print(f"[拓扑] 道路总长度(km)：{total_length_m / 1000:.2f}")
    print(f"[拓扑] 耗时(s)：{elapsed_s:.1f}")
    print(f"[拓扑] 路网图输出：{config.road_graph_path}")
    print(f"[拓扑] 节点坐标输出：{config.node_position_path}")

    export_edges(graph, config)
    return pos


def export_edges(graph: nx.DiGraph, config: AppConfig) -> pd.DataFrame:
    """导出边信息表，字段含义与 notebook 对齐。"""
    rows = []
    for start_node, end_node, attr in graph.edges(data=True):
        kind = int(attr.get("kind", 0))
        rows.append(
            {
                "start_node": start_node,
                "end_node": end_node,
                "kind": kind,
                "duration": float(attr.get("duration", 0.0)),
                "road_length": float(attr.get("length", 0.0)),
                "road_type": KIND_TO_NAME.get(kind, "未知"),
            }
        )

    edges_df = pd.DataFrame(rows)
    edges_df.to_csv(config.edges_csv_path, index=False, encoding="utf-8-sig")
    print(f"[拓扑] 边信息输出：{config.edges_csv_path}（{len(edges_df)} 条）")
    return edges_df


def load_positions(config: AppConfig) -> Dict[str, Tuple[float, float]]:
    """读取节点坐标 npy。若不存在则提示先执行拓扑构建。"""
    validate_input_file(config.node_position_path, "节点坐标npy")
    pos = np.load(config.node_position_path, allow_pickle=True).item()
    if not isinstance(pos, dict):
        raise ValueError(f"节点坐标文件格式不符合预期：{config.node_position_path}")
    return pos


def build_position_dataframe(pos: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """将节点字典转为 DataFrame，便于后续范围筛选。"""
    rows = []
    for node_name, coord in pos.items():
        if len(coord) < 2:
            continue
        lon = float(coord[0])
        lat = float(coord[1])
        rows.append({"pos_name": node_name, "pos_coords": (lon, lat), "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def calculate_candidate_distances(
    target_lon: float,
    target_lat: float,
    candidate_df: pd.DataFrame,
) -> List[Tuple[str, float]]:
    """计算目标点到候选节点的距离，并按距离升序返回。"""
    candidates: List[Tuple[str, float]] = []
    for row in candidate_df.itertuples(index=False):
        dist_m = geodesic((row.lat, row.lon), (target_lat, target_lon)).m
        candidates.append((row.pos_name, dist_m))
    candidates.sort(key=lambda x: x[1])
    return candidates


def match_bayonets(config: AppConfig, pos: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """执行卡口与节点匹配，并输出 csv / npy。"""
    validate_input_file(config.bayonet_shp_path, "卡口shp")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    bayonets = gpd.read_file(config.bayonet_shp_path)
    pos_df = build_position_dataframe(pos)

    if pos_df.empty:
        raise ValueError("节点坐标为空，无法执行卡口匹配。")

    used_nodes = set()
    matched_rows: List[Dict[str, object]] = []
    threshold_m = config.match_threshold_m
    lon_delta = config.lon_degree_per_meter * threshold_m
    lat_delta = config.lat_degree_per_meter * threshold_m

    print(f"[匹配] 开始处理卡口记录：{len(bayonets)}，阈值：{threshold_m}m")

    for idx, bayonet in bayonets.iterrows():
        bayonet_id = str(bayonet.get(config.bayonet_id_field, idx))
        geom = bayonet.geometry

        if not isinstance(geom, Point):
            print(f"[匹配] {idx}-{bayonet_id} 跳过：几何类型 {geometry_type_name(geom)} 不是 Point")
            continue

        target_lon = float(geom.x)
        target_lat = float(geom.y)

        # 第一步：经纬度窗口粗筛，减少后续 geodesic 次数。
        subset = pos_df[
            (pos_df["lat"] >= target_lat - lat_delta)
            & (pos_df["lat"] <= target_lat + lat_delta)
            & (pos_df["lon"] >= target_lon - lon_delta)
            & (pos_df["lon"] <= target_lon + lon_delta)
        ]
        if subset.empty:
            if config.verbose:
                print(f"[匹配] {idx}-{bayonet_id} 失败：窗口内无候选节点")
            continue

        # 第二步：精确测距并按距离排序。
        sorted_candidates = calculate_candidate_distances(
            target_lon=target_lon,
            target_lat=target_lat,
            candidate_df=subset,
        )
        if not sorted_candidates:
            if config.verbose:
                print(f"[匹配] {idx}-{bayonet_id} 失败：无距离候选")
            continue

        # 第三步：一对一匹配。按距离从近到远，选第一个未被占用且在阈值内的节点。
        selected_node: Optional[str] = None
        selected_dist: Optional[float] = None
        for node_name, dist_m in sorted_candidates:
            if dist_m > threshold_m:
                break
            if node_name in used_nodes:
                continue
            selected_node = node_name
            selected_dist = round(dist_m, 2)
            break

        if selected_node is None:
            if config.verbose:
                min_dist = sorted_candidates[0][1]
                print(f"[匹配] {idx}-{bayonet_id} 失败：最近距离 {min_dist:.2f}m 或候选节点已占用")
            continue

        used_nodes.add(selected_node)
        matched_rows.append(
            {
                "node": selected_node,
                config.bayonet_id_field: bayonet_id,
                "dist_m": selected_dist,
            }
        )
        if config.verbose:
            print(f"[匹配] {idx}-{bayonet_id} 成功：{selected_dist}m -> {selected_node}")

    result_df = pd.DataFrame(matched_rows)
    result_df.to_csv(config.match_csv_path, index=False, encoding="utf-8-sig")

    # 按原 notebook 结构保存 npy： [node_list, bayonet_id_list, dist_list]
    node_list = result_df["node"].tolist() if not result_df.empty else []
    id_list = result_df[config.bayonet_id_field].tolist() if not result_df.empty else []
    dist_list = result_df["dist_m"].tolist() if not result_df.empty else []
    np.save(config.match_npy_path, [node_list, id_list, dist_list])

    total_bayonets = len(bayonets)
    total_nodes = len(pos_df)
    matched_count = len(result_df)
    match_rate = (matched_count / total_bayonets) if total_bayonets else 0.0
    real_node_ratio = (matched_count / total_nodes) if total_nodes else 0.0
    virtual_node_count = total_nodes - matched_count
    virtual_node_ratio = (virtual_node_count / total_nodes) if total_nodes else 0.0

    print(f"[匹配] 实际卡口设备数量：{total_bayonets}")
    print(f"[匹配] 路网节点总数量：{total_nodes}")
    print(f"[匹配] 匹配成功率：{match_rate:.2%}")
    print(f"[匹配] 真实卡口节点数：{matched_count}（占比 {real_node_ratio:.2%}）")
    print(f"[匹配] 虚拟节点数：{virtual_node_count}（占比 {virtual_node_ratio:.2%}）")
    print(f"[匹配] 匹配结果输出：{config.match_csv_path}")
    print(f"[匹配] 匹配数组输出：{config.match_npy_path}")
    return result_df


def parse_args() -> argparse.Namespace:
    """命令行参数解析。"""
    parser = argparse.ArgumentParser(description="路网拓扑构建 + 卡口节点匹配")
    parser.add_argument(
        "--mode",
        choices=["all", "topology", "match"],
        default="all",
        help="运行模式：all=全部流程，topology=仅构建拓扑，match=仅做匹配",
    )
    parser.add_argument("--road-shp", type=str, help="覆盖配置中的路网 shp 路径")
    parser.add_argument("--bayonet-shp", type=str, help="覆盖配置中的卡口 shp 路径")
    parser.add_argument("--output-dir", type=str, help="覆盖配置中的输出目录")
    parser.add_argument("--match-threshold", type=float, help="覆盖匹配阈值（米）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args)

    print("[配置] road_shp_path:", config.road_shp_path)
    print("[配置] bayonet_shp_path:", config.bayonet_shp_path)
    print("[配置] output_dir:", config.output_dir)
    print("[配置] mode:", args.mode)

    if args.mode in {"all", "topology"}:
        pos = build_topology(config)
    else:
        pos = load_positions(config)

    if args.mode in {"all", "match"}:
        match_bayonets(config, pos)


if __name__ == "__main__":
    main()
