#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 `preprocess.ipynb` 与 `卡口数据统计.ipynb` 重构为一个可直接运行的脚本。

脚本分为三个阶段：
1. 预处理阶段：从原始卡口 CSV 中筛选中心城区数据，完成字段清洗、车辆筛选、本地车牌筛选与卡口匹配。
2. 统计阶段：基于预处理结果计算小时流量分布、相邻卡口行程时间分布。
3. 补算阶段：只提取“被三重规则排除”的样本，并为其重建 CLLX/RLLX（优先查车辆表，表外按比例随机分配）。

使用方法（默认执行全部流程）：
    python bayonet_data_pipeline.py

只执行统计阶段（复用已有预处理结果）：
    python bayonet_data_pipeline.py --skip-preprocess

只生成“被排除样本补算数据”（不重复计算已纳入样本）：
    python bayonet_data_pipeline.py --supplement-excluded-only

导出配置模板并按外部配置运行：
    python bayonet_data_pipeline.py --dump-config-template ./bayonet_config.json
    python bayonet_data_pipeline.py --config ./bayonet_config.json --supplement-excluded-only
"""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


@dataclass(frozen=True)
class PipelineConfig:
    """集中管理脚本配置，避免在业务逻辑中出现硬编码。"""

    # ---------- 输入文件 ----------
    # 中心城区卡口点位（shp）
    bayonet_shp_path: Path
    # 原始过车数据（超大 CSV）
    raw_bayonet_csv_path: Path
    # 与路网拓扑节点匹配成功的卡口表（CSV，至少包含 KKBH）
    route_matched_bayonet_csv_path: Path

    # ---------- 输出文件 ----------
    # 预处理后记录（后续统计直接读取此文件）
    preprocessed_data_path: Path
    # 规则内样本车辆属性汇总表（由本脚本预处理生成）
    vehicle_info_path: Path
    # 小时流量统计结果
    hourly_stat_path: Path
    # 相邻卡口间隔时间分布统计结果
    travel_time_stat_path: Path
    # 被排除样本补算数据（仅补样本，不含已纳入计算样本）
    excluded_supplement_data_path: Path
    # 被排除样本对应的“设备级”车辆属性分配明细（用于补算）
    excluded_device_attr_path: Path
    # 外部输入的 2021 车辆信息表（用于优先匹配 CLLX/RLLX，表外设备再采样）
    vehicle_table_2021_path: Path

    # ---------- 读取与写出选项 ----------
    raw_csv_chunksize: int = 500_000
    raw_csv_encoding: str | None = None
    output_csv_encoding: str = "gbk"
    # 并行参数：
    # parallel_workers=0 表示自动使用 (CPU核数-2)；
    # parallel_workers=1 表示关闭并行（串行）。
    parallel_workers: int = 0
    # 并行处理中允许同时在飞的 chunk 任务数（过大可能占用更多内存）
    parallel_submit_window: int = 8
    # 监控输出开关：打印阶段耗时、分块吞吐与资源信息
    monitor_enabled: bool = True
    # 是否输出资源信息（CPU%、内存）
    monitor_show_resources: bool = True
    # 每处理多少个 chunk 输出一条监控日志
    monitor_every_n_chunks: int = 1
    random_seed: int = 20260224
    # 表外设备属性采样方式：joint=联合分布采样，independent=边际分布独立采样
    sampling_mode: str = "joint"
    # 采样分布统计是否启用缓存（命中后可跳过整表扫描）
    sampling_dist_cache_enabled: bool = True
    # 是否强制重建采样分布（忽略缓存）
    sampling_dist_force_rebuild: bool = False
    # 采样分布统计并行 worker（<=0 自动取 CPU-2；=1 串行）
    sampling_dist_parallel_workers: int = 0
    # 采样分布分桶数（建议 >= worker 数，便于负载均衡）
    sampling_dist_partition_count: int = 256
    # 是否保留采样分布中间分桶文件（默认清理）
    sampling_dist_keep_temp: bool = False
    # 并行分桶失败时是否自动回退串行构建
    sampling_dist_fallback_serial_on_error: bool = True

    # ---------- 筛选规则 ----------
    valid_vehicle_types: Tuple[str, ...] = ("K1", "K2", "K3", "K4", "H1", "H2", "H3", "H4")
    car_vehicle_types: Tuple[str, ...] = ("K1", "K2", "K3", "K4")
    truck_vehicle_types: Tuple[str, ...] = ("H1", "H2", "H3", "H4")
    valid_fuel_types: Tuple[str, ...] = ("A", "B")  # A: 汽油, B: 柴油
    local_plate_prefixes: Tuple[str, ...] = ("粤E", "粤X", "粤Y")

    # ---------- 行程时间分箱 ----------
    # 分箱边界：与 notebook 逻辑一致，单位秒（左开右闭）
    travel_time_bins: Tuple[int, ...] = (0, 60, 300, 900, 1800, 3600, 28800, 43200, 86400)
    # 分箱标签：需要与 bins 的区间数量一致（len(bins) - 1）
    travel_time_labels: Tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "8h", "12h", "24h")


# ========================== 显式配置区（请按本机路径修改） ==========================
CONFIG = PipelineConfig(
    # 输入
    bayonet_shp_path=Path(
        r"/XYFS01/sysu_yhliu_2/zjt/应用案例分析/佛山行政区和路网数据/中心城区路网和卡口/中心城区卡口点位.shp"
    ),
    raw_bayonet_csv_path=Path(r"/XYFS01/sysu_yhliu_2/zjt/佛山市卡口数据集/2022-06-18_24卡口数据.csv"),
    route_matched_bayonet_csv_path=Path(
        r"/XYFS01/sysu_yhliu_2/zjtdata/路网结构预处理结果/center_realpos.csv"
    ),
    # 输出
    preprocessed_data_path=Path(r"/XYFS01/sysu_yhliu_2/zjtdata/卡口数据预处理_补集/center_valid_df.parquet"),
    vehicle_info_path=Path(r"/XYFS01/sysu_yhliu_2/zjtdata/卡口数据预处理_补集/中心城区研究车辆信息表_规则内样本.parquet"),
    hourly_stat_path=Path(r"/XYFS01/sysu_yhliu_2/zjtdata/卡口数据预处理_补集/中心城区小时卡口数据量.xlsx"),
    travel_time_stat_path=Path(r"/XYFS01/sysu_yhliu_2/zjtdata/卡口数据预处理_补集/中心城区原始相邻卡口间隔时间分布.xlsx"),
    excluded_supplement_data_path=Path(r"/XYFS01/sysu_yhliu_2/zjtdata/卡口数据预处理_补集/center_excluded_supplement_df.parquet"),
    excluded_device_attr_path=Path(r"/XYFS01/sysu_yhliu_2/zjtdata/卡口数据预处理_补集/中心城区补算设备属性分配明细.parquet"),
    vehicle_table_2021_path=Path(r"/XYFS01/sysu_yhliu_2/zjt/应用案例分析/车辆路径重构/2.重构车辆路径/卡口数据预处理/中心城区研究车辆信息表.csv"),
)
# ================================================================================

CLLX_TO_VEH_TYPE = {
    "K1": "大型客车",
    "K2": "中型客车",
    "K3": "小型客车",
    "K4": "微型客车",
    "H1": "重型货车",
    "H2": "中型货车",
    "H3": "轻型货车",
    "H4": "微型货车",
}
RLLX_TO_FUEL_TYPE = {"A": "汽油", "B": "柴油"}
VEH_TYPE_TO_CLLX = {v: k for k, v in CLLX_TO_VEH_TYPE.items()}
FUEL_TYPE_TO_RLLX = {v: k for k, v in RLLX_TO_FUEL_TYPE.items()}


def config_to_jsonable(cfg: PipelineConfig) -> dict[str, Any]:
    """将配置对象转为可写入 JSON 的普通字典。"""
    out: dict[str, Any] = {}
    for f in fields(PipelineConfig):
        value = getattr(cfg, f.name)
        if isinstance(value, Path):
            out[f.name] = str(value)
        elif isinstance(value, tuple):
            out[f.name] = list(value)
        else:
            out[f.name] = value
    return out


def build_config_from_mapping(base_cfg: PipelineConfig, patch: dict[str, Any]) -> PipelineConfig:
    """基于默认配置 + patch 构造新配置，支持 Path/tuple 自动转换。"""
    merged = config_to_jsonable(base_cfg)
    merged.update(patch or {})

    kwargs: dict[str, Any] = {}
    for f in fields(PipelineConfig):
        if f.name not in merged:
            raise ValueError(f"配置缺少字段: {f.name}")
        default_value = getattr(base_cfg, f.name)
        raw = merged[f.name]
        if isinstance(default_value, Path):
            kwargs[f.name] = Path(raw)
        elif isinstance(default_value, tuple):
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"字段 {f.name} 需要列表/元组，当前为: {type(raw)}")
            kwargs[f.name] = tuple(raw)
        else:
            kwargs[f.name] = raw

    return PipelineConfig(**kwargs)


def load_config_from_json(config_path: Path, base_cfg: PipelineConfig) -> PipelineConfig:
    """从 JSON 文件加载配置，并覆盖默认配置。"""
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件根节点必须是对象: {config_path}")

    cfg = build_config_from_mapping(base_cfg, payload)
    print(f"[配置] 已加载外部配置: {config_path}")
    return cfg


def dump_config_template(template_path: Path, base_cfg: PipelineConfig) -> None:
    """导出一份 JSON 配置模板，便于显式维护路径与规则。"""
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(
        json.dumps(config_to_jsonable(base_cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[配置] 已导出配置模板: {template_path}")


def _resource_snapshot(show: bool) -> str:
    """获取资源快照文本（可选）。"""
    if not show or psutil is None:
        return ""
    try:
        proc = psutil.Process(os.getpid())
        rss_gb = proc.memory_info().rss / (1024**3)
        cpu_all = psutil.cpu_percent(interval=None)
        return f"cpu%={cpu_all:.1f}, rss_gb={rss_gb:.2f}"
    except Exception:
        return ""


def log_monitor(tag: str, message: str, enabled: bool, show_resource: bool) -> None:
    """统一监控日志输出格式。"""
    if not enabled:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    res = _resource_snapshot(show_resource)
    tail = f" | {res}" if res else ""
    print(f"[MONITOR][{ts}][{tag}] {message}{tail}")


def print_config_summary(cfg: PipelineConfig) -> None:
    """打印关键配置，便于任务运行时追踪。"""
    print("\n========== 运行配置 ==========")
    print(f"raw_bayonet_csv_path: {cfg.raw_bayonet_csv_path}")
    print(f"bayonet_shp_path: {cfg.bayonet_shp_path}")
    print(f"route_matched_bayonet_csv_path: {cfg.route_matched_bayonet_csv_path}")
    print(f"vehicle_table_2021_path: {cfg.vehicle_table_2021_path}")
    print(f"preprocessed_data_path(parquet): {resolve_parquet_path(cfg.preprocessed_data_path)}")
    print(f"vehicle_info_path(parquet): {resolve_parquet_path(cfg.vehicle_info_path)}")
    print(f"excluded_supplement_data_path(parquet): {resolve_parquet_path(cfg.excluded_supplement_data_path)}")
    print(f"excluded_device_attr_path(parquet): {resolve_parquet_path(cfg.excluded_device_attr_path)}")
    print(
        "rule: "
        f"CLLX={list(cfg.valid_vehicle_types)}, "
        f"RLLX={list(cfg.valid_fuel_types)}, "
        f"FZJG={list(cfg.local_plate_prefixes)}"
    )
    print(
        "parallel: "
        f"workers={resolve_parallel_workers(cfg.parallel_workers)}, "
        f"submit_window={cfg.parallel_submit_window}, "
        f"chunksize={cfg.raw_csv_chunksize}"
    )
    print(
        "sampling: "
        f"mode={cfg.sampling_mode}, "
        f"seed={cfg.random_seed}"
    )
    print(
        "sampling_dist: "
        f"cache_enabled={cfg.sampling_dist_cache_enabled}, "
        f"force_rebuild={cfg.sampling_dist_force_rebuild}, "
        f"workers={resolve_parallel_workers(cfg.sampling_dist_parallel_workers)}, "
        f"partitions={cfg.sampling_dist_partition_count}"
    )
    print(
        "monitor: "
        f"enabled={cfg.monitor_enabled}, "
        f"show_resources={cfg.monitor_show_resources}, "
        f"every_n_chunks={cfg.monitor_every_n_chunks}"
    )
    print("==============================\n")


def ensure_parent_dir(path: Path) -> None:
    """确保输出文件的父目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_parquet_path(path: Path) -> Path:
    """Normalize output path to parquet suffix."""
    return path if path.suffix.lower() == ".parquet" else path.with_suffix(".parquet")


def write_parquet_with_hint(df: pd.DataFrame, path: Path, where: str) -> None:
    """Write parquet with a clear dependency hint on failure."""
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        raise RuntimeError(
            f"{where} 写入 Parquet 失败，请确认已安装 pyarrow 或 fastparquet。原始错误: {exc}"
        ) from exc


def read_parquet_with_hint(path: Path, where: str) -> pd.DataFrame:
    """Read parquet with a clear dependency hint on failure."""
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(
            f"{where} 读取 Parquet 失败，请确认已安装 pyarrow 或 fastparquet。原始错误: {exc}"
        ) from exc


def assert_required_columns(df: pd.DataFrame, required_columns: Iterable[str], where: str) -> None:
    """校验 DataFrame 是否包含必要字段，缺失时给出清晰报错。"""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{where} 缺少必要字段: {missing}")


def read_csv_with_fallback(path: Path, encodings: Tuple[str, ...] = ("gbk", "gb18030", "utf-8-sig", "utf-8")) -> pd.DataFrame:
    """按候选编码依次读取 CSV，提升跨文件来源时的兼容性。"""
    if not path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {path}")

    last_error: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"[读取] {path.name} 使用编码: {enc}")
            return df
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"无法读取 CSV 文件: {path}，最后错误: {last_error}")


def resolve_parallel_workers(configured_workers: int) -> int:
    """
    解析并行 worker 数：
    - <=0: 自动使用 (CPU核数 - 2)
    - =1: 串行
    - >1: 按配置值使用
    """
    if configured_workers <= 0:
        cpu = os.cpu_count() or 1
        return max(1, cpu - 2)
    return max(1, configured_workers)


def process_center_chunk(
    chunk: pd.DataFrame,
    bayonet_ids: Tuple[str, ...],
    required_columns: Tuple[str, ...],
) -> Tuple[pd.DataFrame, int, int]:
    """
    单个分块的标准清洗 + 中心城区过滤。

    该函数会在并行进程中执行，因此需要保持纯函数风格（不依赖外部状态）。
    """
    missing = [col for col in required_columns if col not in chunk.columns]
    if missing:
        raise ValueError(f"原始卡口数据分块缺少必要字段: {missing}")

    # 先清掉关键字段缺失数据，避免后续字符串处理报错
    x = chunk.dropna(subset=["GCRQ", "GCSJ", "KKBH", "FZJG", "CLTMBH"]).copy()

    # 按 notebook 逻辑保留车辆类型前两位（如 K3、H1）
    x["CLLX"] = x["CLLX"].astype(str).str[:2]
    x["KKBH"] = x["KKBH"].astype(str).str.strip()

    total_rows = len(x)
    filtered = x[x["KKBH"].isin(bayonet_ids)]
    matched_rows = len(filtered)
    return filtered, total_rows, matched_rows


def normalize_records_for_split(data: pd.DataFrame) -> pd.DataFrame:
    """
    对中心城区卡口记录做统一清洗（不做三重规则过滤）。

    说明：
    - 该步骤会保留“将来会被排除”的样本，因为补算阶段正是要处理它们。
    - 仅做字段标准化、时间解析、去重与排序，保证后续切分稳定。
    """
    required = ["GCRQ", "GCSJ", "KKBH", "FZJG", "CLTMBH", "CLLX", "RLLX"]
    assert_required_columns(data, required, "中心城区记录")

    x = data.copy()
    x["CLTMBH"] = x["CLTMBH"].astype(str).str.strip()
    x["KKBH"] = x["KKBH"].astype(str).str.strip()
    x["FZJG"] = x["FZJG"].astype(str).str.strip()
    x["CLLX"] = x["CLLX"].astype(str).str.strip().str[:2]
    x["RLLX"] = x["RLLX"].astype(str).str.strip()
    x["GCRQ"] = x["GCRQ"].astype(str).str.split(" ", n=1, expand=True)[0]
    x["GCSJ"] = pd.to_datetime(x["GCSJ"], errors="coerce")

    # 这里不依赖 CLLX/RLLX/FZJG 非空，因为被排除样本可能正是这些字段异常。
    x = x.dropna(subset=["GCRQ", "GCSJ", "CLTMBH", "KKBH"])
    x = x.drop_duplicates(subset=["GCRQ", "GCSJ", "CLTMBH"])
    x = x.sort_values(by=["GCRQ", "GCSJ", "CLTMBH"]).reset_index(drop=True)
    return x


def split_included_and_excluded_records(data: pd.DataFrame, config: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按三重规则切分“已纳入样本”和“被排除样本”。

    注意：补算阶段只使用 excluded，不会重复计算 included。
    """
    cllx_ok = data["CLLX"].isin(config.valid_vehicle_types)
    rllx_ok = data["RLLX"].isin(config.valid_fuel_types)
    fzjg_ok = data["FZJG"].isin(config.local_plate_prefixes)
    in_rule = cllx_ok & rllx_ok & fzjg_ok

    included = data[in_rule].copy().reset_index(drop=True)
    excluded = data[~in_rule].copy().reset_index(drop=True)

    # 为被排除样本标记“被哪条规则排除”，便于后续核查。
    excluded["excluded_by_cllx"] = (~cllx_ok[~in_rule]).to_numpy()
    excluded["excluded_by_rllx"] = (~rllx_ok[~in_rule]).to_numpy()
    excluded["excluded_by_fzjg"] = (~fzjg_ok[~in_rule]).to_numpy()

    reason = np.where(excluded["excluded_by_cllx"], "CLLX", "")
    reason = np.where(
        excluded["excluded_by_rllx"],
        np.where(reason == "", "RLLX", np.char.add(reason, "|RLLX")),
        reason,
    )
    reason = np.where(
        excluded["excluded_by_fzjg"],
        np.where(reason == "", "FZJG", np.char.add(reason, "|FZJG")),
        reason,
    )
    excluded["excluded_reason"] = reason

    print(f"[切分] 已纳入样本记录数: {len(included)}")
    print(f"[切分] 被排除样本记录数: {len(excluded)}")
    return included, excluded


def load_vehicle_table_for_attr_imputation(config: PipelineConfig) -> pd.DataFrame:
    """
    读取 2021 车辆信息表，得到设备级 CLLX/RLLX 样本池。

    优先使用原始编码字段 CLLX/RLLX；若缺失则尝试从 veh_type/fuel_type 反推。
    """
    veh = read_csv_with_fallback(config.vehicle_table_2021_path)
    assert_required_columns(veh, ["CLTMBH"], "2021车辆信息表")

    x = veh.copy()
    x["CLTMBH"] = x["CLTMBH"].astype(str).str.strip()

    if "CLLX" not in x.columns and "veh_type" in x.columns:
        x["CLLX"] = x["veh_type"].astype(str).str.strip().map(VEH_TYPE_TO_CLLX)
    if "RLLX" not in x.columns and "fuel_type" in x.columns:
        x["RLLX"] = x["fuel_type"].astype(str).str.strip().map(FUEL_TYPE_TO_RLLX)

    assert_required_columns(x, ["CLLX", "RLLX"], "2021车辆信息表")
    x["CLLX"] = x["CLLX"].astype(str).str.strip().str[:2]
    x["RLLX"] = x["RLLX"].astype(str).str.strip()

    # 补算赋值池只保留研究口径内的车型与燃料，避免把异常编码采样出去。
    x = x[
        x["CLLX"].isin(config.valid_vehicle_types)
        & x["RLLX"].isin(config.valid_fuel_types)
        & x["CLTMBH"].ne("")
    ].copy()
    x = x.drop_duplicates(subset=["CLTMBH"], keep="first").reset_index(drop=True)

    if x.empty:
        raise ValueError("2021车辆信息表中无可用于采样的有效 CLLX/RLLX 样本。")

    x["veh_type"] = x["CLLX"].map(CLLX_TO_VEH_TYPE)
    x["fuel_type"] = x["RLLX"].map(RLLX_TO_FUEL_TYPE)
    print(f"[补算] 车辆表可用设备数: {x['CLTMBH'].nunique()}")
    return x


def _sampling_cache_paths(config: PipelineConfig) -> tuple[Path, Path]:
    """返回采样分布缓存文件与元数据路径。"""
    base_dir = resolve_parquet_path(config.excluded_supplement_data_path).parent
    cache_path = base_dir / "raw_sampling_distribution_cache.parquet"
    meta_path = base_dir / "raw_sampling_distribution_cache.meta.json"
    return cache_path, meta_path


def _sampling_cache_signature(config: PipelineConfig) -> dict[str, Any]:
    """缓存签名：仅包含会影响分布结果的关键输入。"""
    stat = config.raw_bayonet_csv_path.stat()
    return {
        "raw_path": str(config.raw_bayonet_csv_path),
        "raw_size": int(stat.st_size),
        "raw_mtime_ns": int(stat.st_mtime_ns),
        "raw_csv_chunksize": int(config.raw_csv_chunksize),
        "raw_csv_encoding": config.raw_csv_encoding,
        "logic_version": 1,
    }


def _load_sampling_dist_cache(config: PipelineConfig) -> pd.DataFrame | None:
    """读取采样分布缓存，未命中返回 None。"""
    if not config.sampling_dist_cache_enabled:
        return None
    if config.sampling_dist_force_rebuild:
        print("[补算] 已启用 sampling_dist_force_rebuild，跳过采样分布缓存。")
        return None

    cache_path, meta_path = _sampling_cache_paths(config)
    if not cache_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("signature") != _sampling_cache_signature(config):
            print("[补算] 采样分布缓存签名不匹配，准备重建。")
            return None

        dist = read_parquet_with_hint(cache_path, "采样分布缓存")
        assert_required_columns(dist, ["CLLX", "RLLX", "count"], "采样分布缓存")
        if dist.empty:
            return None
        dist = dist[["CLLX", "RLLX", "count"]].copy()
        dist["count"] = pd.to_numeric(dist["count"], errors="coerce").fillna(0).astype("int64")
        dist = dist[dist["count"] > 0].copy()
        if dist.empty:
            return None
        dist = dist.sort_values(by="count", ascending=False, ignore_index=True)
        dist["p"] = dist["count"] / dist["count"].sum()
        print(f"[补算] 命中采样分布缓存: {cache_path}")
        return dist
    except Exception as exc:
        print(f"[补算] 采样分布缓存读取失败，改为重建。原因: {exc}")
        return None


def _save_sampling_dist_cache(config: PipelineConfig, dist: pd.DataFrame) -> None:
    """写入采样分布缓存。"""
    if not config.sampling_dist_cache_enabled:
        return

    cache_path, meta_path = _sampling_cache_paths(config)
    ensure_parent_dir(cache_path)
    write_parquet_with_hint(dist[["CLLX", "RLLX", "count", "p"]], cache_path, "采样分布缓存")

    payload = {
        "signature": _sampling_cache_signature(config),
        "rows": int(len(dist)),
        "total_count": int(dist["count"].sum()) if not dist.empty else 0,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[补算] 采样分布缓存已更新: {cache_path}")


def _clean_sampling_chunk(chunk: pd.DataFrame, required: tuple[str, ...]) -> pd.DataFrame:
    """按既有逻辑清洗用于采样分布统计的分块。"""
    missing = [col for col in required if col not in chunk.columns]
    if missing:
        raise ValueError(f"原始卡口数据缺少用于采样分布统计的字段: {missing}")

    x = chunk.dropna(subset=list(required)).copy()
    if x.empty:
        return x

    x["CLTMBH"] = x["CLTMBH"].astype(str).str.strip()
    x["CLLX"] = x["CLLX"].astype(str).str.strip().str[:2]
    x["RLLX"] = x["RLLX"].astype(str).str.strip()
    x["GCRQ"] = x["GCRQ"].astype(str).str.split(" ", n=1, expand=True)[0]
    x["GCSJ"] = pd.to_datetime(x["GCSJ"], errors="coerce")
    x = x.dropna(subset=["GCSJ"])
    x = x[
        x["CLTMBH"].ne("")
        & x["CLLX"].ne("")
        & x["RLLX"].ne("")
        & x["GCRQ"].ne("")
        & x["CLTMBH"].str.lower().ne("nan")
        & x["CLLX"].str.lower().ne("nan")
        & x["RLLX"].str.lower().ne("nan")
        & x["GCRQ"].str.lower().ne("nan")
    ]
    if x.empty:
        return x
    return x[["GCRQ", "GCSJ", "CLTMBH", "CLLX", "RLLX"]].copy()


def _build_raw_sampling_distribution_serial(config: PipelineConfig) -> pd.DataFrame:
    """串行构建采样分布（回退路径，口径与原实现一致）。"""
    required = ("GCRQ", "GCSJ", "CLTMBH", "CLLX", "RLLX")
    reader = pd.read_csv(
        config.raw_bayonet_csv_path,
        chunksize=config.raw_csv_chunksize,
        encoding=config.raw_csv_encoding,
        low_memory=False,
    )

    seen_hashes: set[int] = set()
    joint_count: dict[tuple[str, str], int] = {}
    cleaned_rows = 0
    dedup_rows = 0
    monitor_every = max(1, int(config.monitor_every_n_chunks))
    t_start = time.perf_counter()

    for idx, chunk in enumerate(reader, start=1):
        x = _clean_sampling_chunk(chunk, required)
        if x.empty:
            continue

        cleaned_rows += len(x)
        x = x.drop_duplicates(subset=["GCRQ", "GCSJ", "CLTMBH"], keep="first")

        key_hash = pd.util.hash_pandas_object(x[["GCRQ", "GCSJ", "CLTMBH"]], index=False).astype("uint64")
        is_new = ~key_hash.isin(seen_hashes)
        if not is_new.any():
            continue

        seen_hashes.update(int(v) for v in key_hash[is_new].to_list())
        x_new = x.loc[is_new, ["CLLX", "RLLX"]]
        dedup_rows += len(x_new)

        chunk_joint = x_new.value_counts().to_dict()
        for (cllx, rllx), cnt in chunk_joint.items():
            key = (str(cllx), str(rllx))
            joint_count[key] = joint_count.get(key, 0) + int(cnt)

        if idx % monitor_every == 0:
            elapsed = max(time.perf_counter() - t_start, 1e-9)
            log_monitor(
                "sampling-dist",
                (
                    f"mode=serial, chunks={idx}, cleaned_rows={cleaned_rows}, dedup_rows={dedup_rows}, "
                    f"unique_keys={len(seen_hashes)}, rows_per_sec={cleaned_rows / elapsed:.1f}"
                ),
                enabled=config.monitor_enabled,
                show_resource=config.monitor_show_resources,
            )

    if not joint_count:
        raise ValueError("原始 CSV 无法构建采样分布：去重后无有效 CLLX/RLLX 样本。")

    dist = pd.DataFrame(
        [(k[0], k[1], v) for k, v in joint_count.items()],
        columns=["CLLX", "RLLX", "count"],
    ).sort_values(by="count", ascending=False, ignore_index=True)
    dist["p"] = dist["count"] / dist["count"].sum()
    print(
        f"[补算] 原始CSV采样分布构建完成(串行)：联合类别数={len(dist)}，"
        f"去重后样本数={int(dist['count'].sum())}"
    )
    return dist


def _aggregate_sampling_bucket(bucket_path_str: str) -> tuple[int, dict[tuple[str, str], int]]:
    """聚合单个分桶：按最早行序号去重后统计 (CLLX, RLLX) 计数。"""
    bucket_path = Path(bucket_path_str)
    if not bucket_path.exists():
        return 0, {}

    df = pd.read_parquet(bucket_path, columns=["GCRQ", "GCSJ", "CLTMBH", "CLLX", "RLLX", "__row_id"])
    if df.empty:
        return 0, {}

    # 严格保持 keep='first' 语义：按全局行序排序后去重。
    df = df.sort_values(by="__row_id", kind="mergesort")
    dedup = df.drop_duplicates(subset=["GCRQ", "GCSJ", "CLTMBH"], keep="first")
    dedup_rows = int(len(dedup))

    joint: dict[tuple[str, str], int] = {}
    vc = dedup[["CLLX", "RLLX"]].value_counts().to_dict()
    for (cllx, rllx), cnt in vc.items():
        joint[(str(cllx), str(rllx))] = int(cnt)
    return dedup_rows, joint


def _build_raw_sampling_distribution_parallel(config: PipelineConfig) -> pd.DataFrame:
    """并行构建采样分布：分桶落盘 + 多进程聚合，保持原口径与去重语义。"""
    required = ("GCRQ", "GCSJ", "CLTMBH", "CLLX", "RLLX")
    partition_count = max(2, int(config.sampling_dist_partition_count))
    workers = resolve_parallel_workers(config.sampling_dist_parallel_workers)
    monitor_every = max(1, int(config.monitor_every_n_chunks))

    tmp_root = resolve_parquet_path(config.excluded_supplement_data_path).parent / "_sampling_dist_tmp"
    run_dir = tmp_root / f"run_{int(time.time())}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cleaned_rows = 0
    t_partition_start = time.perf_counter()

    try:
        reader = pd.read_csv(
            config.raw_bayonet_csv_path,
            chunksize=config.raw_csv_chunksize,
            encoding=config.raw_csv_encoding,
            low_memory=False,
        )

        for idx, chunk in enumerate(reader, start=1):
            x = _clean_sampling_chunk(chunk, required)
            if x.empty:
                continue

            row_ids = np.arange(cleaned_rows, cleaned_rows + len(x), dtype=np.int64)
            x["__row_id"] = row_ids
            cleaned_rows += len(x)

            key_hash = pd.util.hash_pandas_object(x[["GCRQ", "GCSJ", "CLTMBH"]], index=False).astype("uint64")
            x["__bucket"] = (key_hash % np.uint64(partition_count)).astype("int32").to_numpy()

            for bucket_id, bucket_df in x.groupby("__bucket", sort=False):
                out_dir = run_dir / f"bucket={int(bucket_id)}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"chunk_{idx:06d}.parquet"
                write_parquet_with_hint(bucket_df.drop(columns=["__bucket"]), out_path, "采样分布中间分桶")

            if idx % monitor_every == 0:
                elapsed = max(time.perf_counter() - t_partition_start, 1e-9)
                log_monitor(
                    "sampling-dist",
                    f"phase=partition, chunks={idx}, cleaned_rows={cleaned_rows}, rows_per_sec={cleaned_rows / elapsed:.1f}",
                    enabled=config.monitor_enabled,
                    show_resource=config.monitor_show_resources,
                )

        bucket_dirs = [run_dir / f"bucket={i}" for i in range(partition_count) if (run_dir / f"bucket={i}").exists()]
        if not bucket_dirs:
            raise ValueError("原始 CSV 无法构建采样分布：清洗后无有效样本。")

        joint_count: dict[tuple[str, str], int] = {}
        dedup_rows = 0
        done_buckets = 0
        t_agg_start = time.perf_counter()

        if workers == 1:
            for bucket_dir in bucket_dirs:
                part_rows, part_joint = _aggregate_sampling_bucket(str(bucket_dir))
                dedup_rows += part_rows
                for key, cnt in part_joint.items():
                    joint_count[key] = joint_count.get(key, 0) + cnt
                done_buckets += 1
                if done_buckets % monitor_every == 0:
                    elapsed = max(time.perf_counter() - t_agg_start, 1e-9)
                    log_monitor(
                        "sampling-dist",
                        (
                            f"phase=aggregate, mode=serial, done_buckets={done_buckets}/{len(bucket_dirs)}, "
                            f"dedup_rows={dedup_rows}, buckets_per_sec={done_buckets / elapsed:.2f}"
                        ),
                        enabled=config.monitor_enabled,
                        show_resource=config.monitor_show_resources,
                    )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_aggregate_sampling_bucket, str(path)): path for path in bucket_dirs}
                for fut in as_completed(futures):
                    part_rows, part_joint = fut.result()
                    dedup_rows += part_rows
                    for key, cnt in part_joint.items():
                        joint_count[key] = joint_count.get(key, 0) + cnt
                    done_buckets += 1
                    if done_buckets % monitor_every == 0:
                        elapsed = max(time.perf_counter() - t_agg_start, 1e-9)
                        log_monitor(
                            "sampling-dist",
                            (
                                f"phase=aggregate, mode=parallel, done_buckets={done_buckets}/{len(bucket_dirs)}, "
                                f"dedup_rows={dedup_rows}, buckets_per_sec={done_buckets / elapsed:.2f}"
                            ),
                            enabled=config.monitor_enabled,
                            show_resource=config.monitor_show_resources,
                        )

        if not joint_count:
            raise ValueError("原始 CSV 无法构建采样分布：去重后无有效 CLLX/RLLX 样本。")

        dist = pd.DataFrame(
            [(k[0], k[1], v) for k, v in joint_count.items()],
            columns=["CLLX", "RLLX", "count"],
        ).sort_values(by="count", ascending=False, ignore_index=True)
        dist["p"] = dist["count"] / dist["count"].sum()
        print(
            f"[补算] 原始CSV采样分布构建完成(并行)：联合类别数={len(dist)}，"
            f"去重后样本数={int(dist['count'].sum())}"
        )
        return dist
    finally:
        if config.sampling_dist_keep_temp:
            print(f"[补算] 保留采样分布中间文件: {run_dir}")
        else:
            shutil.rmtree(run_dir, ignore_errors=True)


def build_raw_sampling_distribution(config: PipelineConfig) -> pd.DataFrame:
    """
    从原始 CSV 构建采样分布：
    - 先按 (GCRQ, GCSJ, CLTMBH) 去重
    - 再统计 (CLLX, RLLX) 的联合分布
    """
    cached = _load_sampling_dist_cache(config)
    if cached is not None:
        return cached

    use_parallel = (
        resolve_parallel_workers(config.sampling_dist_parallel_workers) > 1
        and int(config.sampling_dist_partition_count) > 1
    )

    if use_parallel:
        try:
            dist = _build_raw_sampling_distribution_parallel(config)
        except Exception as exc:
            if not config.sampling_dist_fallback_serial_on_error:
                raise
            print(f"[补算] 并行采样分布构建失败，回退串行。原因: {exc}")
            dist = _build_raw_sampling_distribution_serial(config)
    else:
        dist = _build_raw_sampling_distribution_serial(config)

    _save_sampling_dist_cache(config, dist)
    return dist



def sample_missing_device_attrs(missing_device_ids: pd.Series, sampling_dist: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """对车辆表外设备按比例随机分配 CLLX/RLLX（设备级，保持同设备一致）。"""
    devices = pd.Series(missing_device_ids.astype(str).unique(), name="CLTMBH")
    if devices.empty:
        return pd.DataFrame(columns=["CLTMBH", "CLLX", "RLLX"])

    if sampling_dist.empty:
        raise ValueError("采样分布为空，无法为表外设备分配 CLLX/RLLX。")

    rng = np.random.default_rng(config.random_seed)
    out = pd.DataFrame({"CLTMBH": devices})
    mode = str(config.sampling_mode).strip().lower()

    if mode == "joint":
        # 联合分布采样：一次性采样 (CLLX, RLLX) 组合，保留二者耦合关系。
        pick = rng.choice(len(sampling_dist), size=len(out), p=sampling_dist["p"].to_numpy(dtype=float))
        sampled = sampling_dist.iloc[pick][["CLLX", "RLLX"]].reset_index(drop=True)
        out["CLLX"] = sampled["CLLX"].to_numpy()
        out["RLLX"] = sampled["RLLX"].to_numpy()
    elif mode == "independent":
        # 边际分布独立采样：分别按 CLLX、RLLX 比例采样。
        cllx_prob = sampling_dist.groupby("CLLX", sort=False)["count"].sum()
        cllx_prob = cllx_prob / cllx_prob.sum()
        rllx_prob = sampling_dist.groupby("RLLX", sort=False)["count"].sum()
        rllx_prob = rllx_prob / rllx_prob.sum()
        out["CLLX"] = rng.choice(cllx_prob.index.to_numpy(), size=len(out), p=cllx_prob.to_numpy())
        out["RLLX"] = rng.choice(rllx_prob.index.to_numpy(), size=len(out), p=rllx_prob.to_numpy())
    else:
        raise ValueError(f"sampling_mode 仅支持 joint 或 independent，当前为: {config.sampling_mode}")

    return out


def rebuild_excluded_vehicle_attrs(excluded_df: pd.DataFrame, config: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    为被排除样本重建 CLLX/RLLX：
    1. 设备在车辆表内：直接取表内属性；
    2. 设备不在车辆表：按原始 CSV 去重后统计分布随机分配。
    """
    pool = load_vehicle_table_for_attr_imputation(config)
    sampling_dist = build_raw_sampling_distribution(config)

    devices = pd.DataFrame({"CLTMBH": excluded_df["CLTMBH"].astype(str).unique()})
    known = devices.merge(pool[["CLTMBH", "CLLX", "RLLX"]], on="CLTMBH", how="left")
    known["is_in_vehicle_table"] = known["CLLX"].notna() & known["RLLX"].notna()

    missing_ids = known.loc[~known["is_in_vehicle_table"], "CLTMBH"]
    sampled = sample_missing_device_attrs(missing_ids, sampling_dist, config)
    if not sampled.empty:
        known = known.merge(
            sampled.rename(columns={"CLLX": "CLLX_sampled", "RLLX": "RLLX_sampled"}),
            on="CLTMBH",
            how="left",
        )
        missing_mask = ~known["is_in_vehicle_table"]
        known.loc[missing_mask, "CLLX"] = known.loc[missing_mask, "CLLX_sampled"]
        known.loc[missing_mask, "RLLX"] = known.loc[missing_mask, "RLLX_sampled"]
        known.drop(columns=["CLLX_sampled", "RLLX_sampled"], inplace=True)

    known["assigned_by_sampling"] = ~known["is_in_vehicle_table"]
    known["veh_type"] = known["CLLX"].map(CLLX_TO_VEH_TYPE)
    known["fuel_type"] = known["RLLX"].map(RLLX_TO_FUEL_TYPE)
    known["device_id"] = known["CLTMBH"]

    if known["CLLX"].isna().any() or known["RLLX"].isna().any():
        raise ValueError("存在设备未完成 CLLX/RLLX 重建，请检查车辆表与采样逻辑。")

    # 保留原始 CLLX/RLLX，并将重建值写回 CLLX/RLLX（供后续 BWP 直接使用）
    rebuilt = excluded_df.copy()
    rebuilt["CLTMBH"] = rebuilt["CLTMBH"].astype(str).str.strip()
    rebuilt["device_id"] = rebuilt["CLTMBH"]
    rebuilt = rebuilt.rename(columns={"CLLX": "CLLX_raw", "RLLX": "RLLX_raw"})
    rebuilt = rebuilt.merge(
        known[["CLTMBH", "CLLX", "RLLX", "veh_type", "fuel_type", "is_in_vehicle_table", "assigned_by_sampling"]],
        on="CLTMBH",
        how="left",
    )

    return rebuilt, known


def load_bayonet_base(config: PipelineConfig) -> gpd.GeoDataFrame:
    """读取中心城区卡口点位，并统一 KKBH 字段格式。"""
    bayonets = gpd.read_file(config.bayonet_shp_path)
    assert_required_columns(bayonets, ["KKBH"], "卡口点位文件")
    bayonets["KKBH"] = bayonets["KKBH"].astype(str).str.strip()
    return bayonets


def filter_center_area_records(config: PipelineConfig, bayonet_ids: set[str]) -> pd.DataFrame:
    """
    分块读取原始卡口数据并筛选中心城区卡口记录。

    这样做的目的：
    1. 避免一次性读入超大 CSV 造成内存压力。
    2. 在读取时就完成关键字段清洗，减少后续重复处理。
    """

    required = ("GCRQ", "GCSJ", "KKBH", "FZJG", "CLTMBH", "CLLX", "RLLX")
    t_start = time.perf_counter()

    filtered_chunks: List[pd.DataFrame] = []
    total_rows = 0
    matched_rows = 0

    reader = pd.read_csv(
        config.raw_bayonet_csv_path,
        chunksize=config.raw_csv_chunksize,
        encoding=config.raw_csv_encoding,
        low_memory=False,
    )

    workers = resolve_parallel_workers(config.parallel_workers)
    bayonet_id_tuple = tuple(bayonet_ids)
    monitor_every = max(1, int(config.monitor_every_n_chunks))

    # 串行路径：便于小数据集或调试时使用。
    if workers == 1:
        print("[预处理] 以串行模式处理分块（parallel_workers=1）")
        for idx, chunk in enumerate(reader, start=1):
            chunk_filtered, t_rows, m_rows = process_center_chunk(chunk, bayonet_id_tuple, required)
            total_rows += t_rows
            matched_rows += m_rows
            if not chunk_filtered.empty:
                filtered_chunks.append(chunk_filtered)
            print(f"[预处理] 第 {idx} 块完成，累计匹配中心城区记录: {matched_rows}")
            if idx % monitor_every == 0:
                elapsed = max(time.perf_counter() - t_start, 1e-9)
                speed = total_rows / elapsed
                log_monitor(
                    "chunk",
                    f"mode=serial, done={idx}, total_rows={total_rows}, matched_rows={matched_rows}, rows_per_sec={speed:.1f}",
                    enabled=config.monitor_enabled,
                    show_resource=config.monitor_show_resources,
                )
    else:
        # 并行路径：利用多核处理 chunk 清洗与过滤。
        submit_window = max(1, int(config.parallel_submit_window))
        print(f"[预处理] 并行分块处理开启，workers={workers}，submit_window={submit_window}")

        pending: dict = {}
        completed = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, chunk in enumerate(reader, start=1):
                fut = executor.submit(process_center_chunk, chunk, bayonet_id_tuple, required)
                pending[fut] = idx

                # 控制在飞任务数量，避免内存无限增长。
                while len(pending) >= submit_window:
                    done = next(as_completed(list(pending.keys())))
                    done_idx = pending.pop(done)
                    chunk_filtered, t_rows, m_rows = done.result()
                    total_rows += t_rows
                    matched_rows += m_rows
                    completed += 1
                    if not chunk_filtered.empty:
                        filtered_chunks.append(chunk_filtered)
                    print(
                        f"[预处理] 分块 {done_idx} 完成（并行已完成 {completed} 块），累计匹配中心城区记录: {matched_rows}"
                    )
                    if completed % monitor_every == 0:
                        elapsed = max(time.perf_counter() - t_start, 1e-9)
                        speed = total_rows / elapsed
                        log_monitor(
                            "chunk",
                            (
                                f"mode=parallel, done={completed}, total_rows={total_rows}, "
                                f"matched_rows={matched_rows}, rows_per_sec={speed:.1f}, pending={len(pending)}"
                            ),
                            enabled=config.monitor_enabled,
                            show_resource=config.monitor_show_resources,
                        )

            # 清空剩余任务。
            while pending:
                done = next(as_completed(list(pending.keys())))
                done_idx = pending.pop(done)
                chunk_filtered, t_rows, m_rows = done.result()
                total_rows += t_rows
                matched_rows += m_rows
                completed += 1
                if not chunk_filtered.empty:
                    filtered_chunks.append(chunk_filtered)
                print(
                    f"[预处理] 分块 {done_idx} 完成（并行已完成 {completed} 块），累计匹配中心城区记录: {matched_rows}"
                )
                if completed % monitor_every == 0:
                    elapsed = max(time.perf_counter() - t_start, 1e-9)
                    speed = total_rows / elapsed
                    log_monitor(
                        "chunk",
                        (
                            f"mode=parallel, done={completed}, total_rows={total_rows}, "
                            f"matched_rows={matched_rows}, rows_per_sec={speed:.1f}, pending={len(pending)}"
                        ),
                        enabled=config.monitor_enabled,
                        show_resource=config.monitor_show_resources,
                    )

    if not filtered_chunks:
        raise ValueError("未读取到任何原始数据分块，请检查原始 CSV 文件路径。")

    merged = pd.concat(filtered_chunks, ignore_index=True)
    ratio = matched_rows / total_rows if total_rows else 0.0
    print(f"[预处理] 原始有效记录: {total_rows}")
    print(f"[预处理] 中心城区匹配记录: {len(merged)}，匹配率: {ratio:.2%}")
    elapsed = max(time.perf_counter() - t_start, 1e-9)
    log_monitor(
        "chunk-summary",
        f"elapsed_sec={elapsed:.1f}, total_rows={total_rows}, matched_rows={matched_rows}, rows_per_sec={total_rows / elapsed:.1f}",
        enabled=config.monitor_enabled,
        show_resource=config.monitor_show_resources,
    )
    return merged


def preprocess_valid_vehicle_records(center_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    对中心城区记录进行研究口径清洗：
    1. 仅保留客车/货车（K1-K4, H1-H4）
    2. 仅保留汽油/柴油（A/B）
    3. 仅保留佛山市本地牌照（粤E/粤X/粤Y）
    4. 统一时间字段并去重
    """

    origin_total = len(center_df)

    valid_df = center_df[center_df["CLLX"].isin(config.valid_vehicle_types)]
    valid_df = valid_df[valid_df["RLLX"].isin(config.valid_fuel_types)]
    valid_df = valid_df.dropna(subset=["CLLX"]).copy()
    valid_df.reset_index(drop=True, inplace=True)

    print(
        f"[预处理] 研究车辆原始记录: {len(valid_df)}，占中心城区记录比例: {len(valid_df) / origin_total:.2%}"
        if origin_total
        else "[预处理] 中心城区记录为空，无法继续。"
    )

    # 仅保留佛山市本地车辆
    valid_df = valid_df[valid_df["FZJG"].isin(config.local_plate_prefixes)].copy()
    print(f"[预处理] 佛山市本地车辆记录: {len(valid_df)}")

    # 字段格式化：GCRQ 保留日期部分；GCSJ 转时间戳
    valid_df["GCRQ"] = valid_df["GCRQ"].astype(str).str.split(" ", n=1, expand=True)[0]
    valid_df["GCSJ"] = pd.to_datetime(valid_df["GCSJ"])

    # 去重逻辑沿用 notebook：同一日期+时刻+车牌只保留一条
    valid_df = valid_df.drop_duplicates(subset=["GCRQ", "GCSJ", "CLTMBH"])

    # 先按时间主序排序，保证后续统计逻辑稳定
    valid_df = valid_df.sort_values(by=["GCRQ", "GCSJ", "CLTMBH"]).reset_index(drop=True)
    print(f"[预处理] 清洗后记录数: {len(valid_df)}")

    return valid_df


def add_coordinate_columns(bayonets: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    为卡口点位补充经纬度字段。

    notebook 里使用了 `Longitude` / `Latitude`，但不同数据源字段名可能不一致，
    这里兼容两种情况：
    1. 已有 Longitude/Latitude 字段 -> 直接使用
    2. 仅有 geometry 点位 -> 从几何中提取 x/y 作为经纬度
    """

    if "Longitude" in bayonets.columns and "Latitude" in bayonets.columns:
        return bayonets

    if "geometry" not in bayonets.columns:
        raise ValueError("卡口点位文件既没有 Longitude/Latitude，也没有 geometry 字段。")

    bayonets = bayonets.copy()
    bayonets["Longitude"] = bayonets.geometry.x
    bayonets["Latitude"] = bayonets.geometry.y
    return bayonets


def match_with_topology_nodes(
    valid_df: pd.DataFrame,
    bayonets: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """仅保留能与“真实路网节点匹配卡口表”对齐的数据，并补充经纬度。"""

    pos_bayonets = pd.read_csv(config.route_matched_bayonet_csv_path, low_memory=False)
    assert_required_columns(pos_bayonets, ["KKBH"], "路网匹配卡口文件")
    pos_bayonets["KKBH"] = pos_bayonets["KKBH"].astype(str).str.strip()

    bayonets = add_coordinate_columns(bayonets)

    mapped = pd.merge(valid_df, pos_bayonets[["KKBH"]], on="KKBH", how="inner")
    mapped = pd.merge(mapped, bayonets[["KKBH", "Longitude", "Latitude"]], on="KKBH", how="left")

    ratio = len(mapped) / len(valid_df) if len(valid_df) else 0.0
    print(f"[预处理] 与真实路网卡口匹配成功: {len(mapped)}，匹配率: {ratio:.2%}")

    # 最终排序采用 notebook 的后续顺序：车牌 -> 时间 -> 日期
    mapped = mapped.sort_values(by=["CLTMBH", "GCSJ", "GCRQ"]).reset_index(drop=True)
    return mapped


def build_vehicle_info_table(mapped_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建车辆信息表（每个车牌一条）。
    若部分列不存在，则跳过该列，避免因源数据字段差异导致脚本中断。
    """

    cllx_mapping = {
        "K1": "大型客车",
        "K2": "中型客车",
        "K3": "小型客车",
        "K4": "微型客车",
        "H1": "重型货车",
        "H2": "中型货车",
        "H3": "轻型货车",
        "H4": "微型货车",
    }
    rllx_mapping = {"A": "汽油", "B": "柴油"}

    vehicle_info = mapped_df.drop_duplicates(subset="CLTMBH", keep="first").copy()

    keep_cols = [col for col in ["CLTMBH", "HPZL", "CLLX", "RLLX", "PFBZ", "SYXZ"] if col in vehicle_info.columns]
    vehicle_info = vehicle_info[keep_cols]
    if "CLLX" in vehicle_info.columns:
        vehicle_info["veh_type"] = vehicle_info["CLLX"].map(cllx_mapping)
    if "RLLX" in vehicle_info.columns:
        vehicle_info["fuel_type"] = vehicle_info["RLLX"].map(rllx_mapping)

    print(f"[预处理] 研究车辆（去重后）: {len(vehicle_info)}")
    return vehicle_info


def static_hour_volume(data: pd.DataFrame) -> pd.DataFrame:
    """统计小时流量及占比。"""
    if data.empty:
        return pd.DataFrame({"HOUR": [], "count": [], "perc": []})

    temp = data.copy()
    temp["HOUR"] = temp["GCSJ"].dt.hour
    hour_stat = temp.groupby("HOUR").size().reset_index(name="count")
    hour_stat["perc"] = hour_stat["count"] / hour_stat["count"].sum()
    return hour_stat


def check_time_precision_to_second(data: pd.DataFrame) -> None:
    """
    检查时间字段是否精确到秒。

    notebook 中对该问题做了单独校验，这里保留同样的质量检查逻辑。
    """
    has_ms = data["GCSJ"].dt.microsecond != 0
    unprecise_rows = data[has_ms].head(10)

    if len(unprecise_rows) > 0:
        print("[统计] 发现时间不精确到秒的样本（最多展示 10 条）：")
        print(unprecise_rows[["GCSJ"]])
    else:
        print("[统计] 所有时间均精确到秒。")


def static_travel_time(data: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    统计相邻卡口记录的时间间隔分布。

    口径与 notebook 保持一致：
    - 只比较相邻两条记录
    - 仅当两条记录属于同一车牌、同一天时，才计算时间差
    """

    if data.empty:
        return pd.DataFrame(
            {"travel_time_label": list(config.travel_time_labels), "count": [0] * len(config.travel_time_labels), "perc": [0.0] * len(config.travel_time_labels)}
        )

    # 先按车牌、日期、时间排序，确保“相邻记录”具备时序意义
    ordered = data.sort_values(by=["CLTMBH", "GCRQ", "GCSJ"]).reset_index(drop=True)

    intervals: List[float] = []
    for i in range(len(ordered) - 1):
        same_plate = ordered.loc[i, "CLTMBH"] == ordered.loc[i + 1, "CLTMBH"]
        same_date = ordered.loc[i, "GCRQ"] == ordered.loc[i + 1, "GCRQ"]
        if same_plate and same_date:
            delta = (ordered.loc[i + 1, "GCSJ"] - ordered.loc[i, "GCSJ"]).total_seconds()
            intervals.append(delta)

    if not intervals:
        print("[统计] 未形成可用相邻卡口间隔数据。")
        return pd.DataFrame(
            {"travel_time_label": list(config.travel_time_labels), "count": [0] * len(config.travel_time_labels), "perc": [0.0] * len(config.travel_time_labels)}
        )

    travel_df = pd.DataFrame({"time": intervals})
    print(f"[统计] 最小间隔(s): {travel_df['time'].min()}")
    print(f"[统计] 最大间隔(s): {travel_df['time'].max()}")
    print(f"[统计] 平均间隔(s): {travel_df['time'].mean():.2f}")
    print(f"[统计] 中位数间隔(s): {travel_df['time'].median():.2f}")

    if len(config.travel_time_bins) - 1 != len(config.travel_time_labels):
        raise ValueError("travel_time_bins 与 travel_time_labels 数量不匹配。")

    bucket = pd.cut(
        travel_df["time"],
        bins=list(config.travel_time_bins),
        labels=list(config.travel_time_labels),
    )

    # reindex 保证输出始终按配置标签顺序排列（即使某个区间没有样本）
    counts = bucket.value_counts().reindex(config.travel_time_labels, fill_value=0)
    result = counts.reset_index()
    result.columns = ["travel_time_label", "count"]
    total = result["count"].sum()
    result["perc"] = result["count"] / total if total else 0.0
    return result


def run_preprocess_stage(config: PipelineConfig) -> pd.DataFrame:
    """执行预处理阶段，并输出预处理结果与车辆信息表。"""
    print("\n========== 阶段 1/3：预处理 ==========")
    t_stage = time.perf_counter()
    log_monitor("stage", "start preprocess", config.monitor_enabled, config.monitor_show_resources)
    bayonets = load_bayonet_base(config)
    bayonet_ids = set(bayonets["KKBH"])

    center_df = filter_center_area_records(config, bayonet_ids)
    valid_df = preprocess_valid_vehicle_records(center_df, config)
    mapped_df = match_with_topology_nodes(valid_df, bayonets, config)

    preprocessed_out = resolve_parquet_path(config.preprocessed_data_path)
    ensure_parent_dir(preprocessed_out)
    write_parquet_with_hint(mapped_df, preprocessed_out, "预处理结果")
    print(f"[预处理] 已保存(Parquet): {preprocessed_out}")

    vehicle_info = build_vehicle_info_table(mapped_df)
    vehicle_info_out = resolve_parquet_path(config.vehicle_info_path)
    ensure_parent_dir(vehicle_info_out)
    write_parquet_with_hint(vehicle_info, vehicle_info_out, "车辆信息表")
    print(f"[预处理] 已保存(Parquet): {vehicle_info_out}")
    elapsed = time.perf_counter() - t_stage
    log_monitor(
        "stage",
        f"finish preprocess, elapsed_sec={elapsed:.1f}, records={len(mapped_df)}",
        config.monitor_enabled,
        config.monitor_show_resources,
    )

    return mapped_df


def run_excluded_supplement_stage(config: PipelineConfig) -> pd.DataFrame:
    """
    仅生成“被排除样本补算数据”：
    - 不重复输出已纳入计算样本；
    - 按设备级重建 CLLX/RLLX（先查表、后采样）。
    """
    print("\n========== 阶段 3/3：被排除样本补算 ==========")
    t_stage = time.perf_counter()
    log_monitor("stage", "start supplement-excluded", config.monitor_enabled, config.monitor_show_resources)

    bayonets = load_bayonet_base(config)
    bayonet_ids = set(bayonets["KKBH"])

    center_df = filter_center_area_records(config, bayonet_ids)
    center_df = normalize_records_for_split(center_df)

    _, excluded_df = split_included_and_excluded_records(center_df, config)
    if excluded_df.empty:
        print("[补算] 当前数据中没有被排除样本，无需补算。")
        elapsed = time.perf_counter() - t_stage
        log_monitor(
            "stage",
            f"finish supplement-excluded, elapsed_sec={elapsed:.1f}, records=0",
            config.monitor_enabled,
            config.monitor_show_resources,
        )
        return excluded_df

    # 与旧流程保持一致：仅保留可映射到真实路网节点的卡口记录。
    excluded_mapped = match_with_topology_nodes(excluded_df, bayonets, config)
    if excluded_mapped.empty:
        print("[补算] 被排除样本在路网映射后为空，无可用输出。")
        elapsed = time.perf_counter() - t_stage
        log_monitor(
            "stage",
            f"finish supplement-excluded, elapsed_sec={elapsed:.1f}, records=0_after_map",
            config.monitor_enabled,
            config.monitor_show_resources,
        )
        return excluded_mapped

    rebuilt_df, device_attr_df = rebuild_excluded_vehicle_attrs(excluded_mapped, config)

    excluded_out = resolve_parquet_path(config.excluded_supplement_data_path)
    ensure_parent_dir(excluded_out)
    write_parquet_with_hint(rebuilt_df, excluded_out, "补算结果")
    print(f"[补算] 被排除样本补算数据已保存(Parquet): {excluded_out}")

    device_attr_out = resolve_parquet_path(config.excluded_device_attr_path)
    ensure_parent_dir(device_attr_out)
    write_parquet_with_hint(device_attr_df, device_attr_out, "设备属性分配明细")
    print(f"[补算] 设备级属性分配明细已保存(Parquet): {device_attr_out}")

    sample_device_count = int(device_attr_df["assigned_by_sampling"].sum()) if not device_attr_df.empty else 0
    print(f"[补算] 补算设备总数: {len(device_attr_df)}，其中采样分配设备: {sample_device_count}")
    print(f"[补算] 补算记录数（仅 excluded）: {len(rebuilt_df)}")
    elapsed = time.perf_counter() - t_stage
    log_monitor(
        "stage",
        (
            f"finish supplement-excluded, elapsed_sec={elapsed:.1f}, "
            f"records={len(rebuilt_df)}, devices={len(device_attr_df)}, sampled_devices={sample_device_count}"
        ),
        config.monitor_enabled,
        config.monitor_show_resources,
    )
    return rebuilt_df


def run_statistics_stage(config: PipelineConfig, mapped_df: pd.DataFrame | None = None) -> None:
    """执行统计阶段，输出小时流量和行程时间分布。"""
    print("\n========== 阶段 2/3：统计 ==========")
    t_stage = time.perf_counter()
    log_monitor("stage", "start statistics", config.monitor_enabled, config.monitor_show_resources)

    # 若未传入预处理数据，则从文件读取（便于单独运行统计阶段）
    if mapped_df is None:
        preprocessed_parquet = resolve_parquet_path(config.preprocessed_data_path)
        if preprocessed_parquet.exists():
            mapped_df = read_parquet_with_hint(preprocessed_parquet, "预处理结果")
            print(f"[统计] 读取预处理数据(Parquet): {preprocessed_parquet}")
        elif config.preprocessed_data_path.exists():
            mapped_df = pd.read_csv(config.preprocessed_data_path, encoding=config.output_csv_encoding, low_memory=False)
            print(f"[统计] 未找到 Parquet，回退读取 CSV: {config.preprocessed_data_path}")
        else:
            raise FileNotFoundError(
                f"未找到预处理结果文件: parquet={preprocessed_parquet}, csv={config.preprocessed_data_path}"
            )
        mapped_df["GCSJ"] = pd.to_datetime(mapped_df["GCSJ"])
        mapped_df["KKBH"] = mapped_df["KKBH"].astype(str)

    assert_required_columns(mapped_df, ["GCSJ", "CLLX", "CLTMBH", "GCRQ", "KKBH"], "预处理结果数据")
    check_time_precision_to_second(mapped_df)

    cars_df = mapped_df[mapped_df["CLLX"].isin(config.car_vehicle_types)].reset_index(drop=True)
    trucks_df = mapped_df[mapped_df["CLLX"].isin(config.truck_vehicle_types)].reset_index(drop=True)

    # 小时流量统计
    cars_volume = static_hour_volume(cars_df)
    trucks_volume = static_hour_volume(trucks_df)

    all_hours = pd.DataFrame({"hour": list(range(24))})
    cars_hour = (
        cars_volume.rename(columns={"HOUR": "hour", "count": "car_count", "perc": "car_perc"})[["hour", "car_count", "car_perc"]]
        if not cars_volume.empty
        else pd.DataFrame({"hour": list(range(24)), "car_count": [0] * 24, "car_perc": [0.0] * 24})
    )
    trucks_hour = (
        trucks_volume.rename(columns={"HOUR": "hour", "count": "truck_count", "perc": "truck_perc"})[["hour", "truck_count", "truck_perc"]]
        if not trucks_volume.empty
        else pd.DataFrame({"hour": list(range(24)), "truck_count": [0] * 24, "truck_perc": [0.0] * 24})
    )

    volume_result = all_hours.merge(cars_hour, on="hour", how="left").merge(trucks_hour, on="hour", how="left")
    volume_result[["car_count", "car_perc", "truck_count", "truck_perc"]] = volume_result[
        ["car_count", "car_perc", "truck_count", "truck_perc"]
    ].fillna(0)

    ensure_parent_dir(config.hourly_stat_path)
    volume_result.to_excel(config.hourly_stat_path, index=False)
    print(f"[统计] 小时流量结果已保存: {config.hourly_stat_path}")

    # 相邻卡口时间间隔统计
    print("[统计] 客车间隔统计：")
    car_tt = static_travel_time(cars_df, config)
    print("[统计] 货车间隔统计：")
    truck_tt = static_travel_time(trucks_df, config)

    travel_result = pd.DataFrame(
        {
            "travel_time_label": list(config.travel_time_labels),
            "car_count": car_tt["count"].values,
            "car_perc": car_tt["perc"].values,
            "truck_count": truck_tt["count"].values,
            "truck_perc": truck_tt["perc"].values,
        }
    )

    ensure_parent_dir(config.travel_time_stat_path)
    travel_result.to_excel(config.travel_time_stat_path, index=False)
    print(f"[统计] 相邻卡口间隔结果已保存: {config.travel_time_stat_path}")
    elapsed = time.perf_counter() - t_stage
    log_monitor(
        "stage",
        f"finish statistics, elapsed_sec={elapsed:.1f}, records={len(mapped_df)}",
        config.monitor_enabled,
        config.monitor_show_resources,
    )


def main(
    config: PipelineConfig,
    skip_preprocess: bool = False,
    skip_statistics: bool = False,
    supplement_excluded_only: bool = False,
) -> None:
    """主入口：支持 legacy 流程和“仅补被排除样本”流程。"""
    t_total = time.perf_counter()
    print_config_summary(config)
    log_monitor("pipeline", "pipeline-start", config.monitor_enabled, config.monitor_show_resources)

    if supplement_excluded_only:
        run_excluded_supplement_stage(config)
        print("\n流程执行完成。")
        elapsed = time.perf_counter() - t_total
        log_monitor(
            "pipeline",
            f"pipeline-finish, elapsed_sec={elapsed:.1f}",
            config.monitor_enabled,
            config.monitor_show_resources,
        )
        return

    mapped_df: pd.DataFrame | None = None

    if not skip_preprocess:
        mapped_df = run_preprocess_stage(config)
    else:
        print("[流程] 已跳过预处理阶段，将直接读取预处理文件做统计。")

    if not skip_statistics:
        run_statistics_stage(config, mapped_df=mapped_df)
    else:
        print("[流程] 已跳过统计阶段。")

    print("\n流程执行完成。")
    elapsed = time.perf_counter() - t_total
    log_monitor(
        "pipeline",
        f"pipeline-finish, elapsed_sec={elapsed:.1f}",
        config.monitor_enabled,
        config.monitor_show_resources,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="中心城区卡口数据预处理与统计一体化脚本")
    parser.add_argument("--config", type=str, help="外部 JSON 配置路径（覆盖脚本内 CONFIG）")
    parser.add_argument("--dump-config-template", type=str, help="导出 JSON 配置模板到指定路径后退出")
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过预处理阶段，直接读取预处理文件做统计")
    parser.add_argument("--skip-statistics", action="store_true", help="跳过统计阶段，仅执行预处理")
    parser.add_argument(
        "--supplement-excluded-only",
        action="store_true",
        help="仅补算被三重规则排除的样本（不重复计算已纳入样本）",
    )
    args = parser.parse_args()

    runtime_config = CONFIG
    if args.dump_config_template:
        dump_config_template(Path(args.dump_config_template), CONFIG)
        raise SystemExit(0)
    if args.config:
        runtime_config = load_config_from_json(Path(args.config), CONFIG)

    main(
        config=runtime_config,
        skip_preprocess=args.skip_preprocess,
        skip_statistics=args.skip_statistics,
        supplement_excluded_only=args.supplement_excluded_only,
    )
