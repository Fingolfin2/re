#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory partition pipeline.

Business constraints from user:
1. Input scope is excluded records only.
2. Segmentation semantics must follow legacy notebook logic.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class PartitionConfig:
    # input
    excluded_input_path: Path
    bayonet_match_path: Path

    # output
    output_dir: Path
    output_prefix: str = "excluded_"

    # io
    input_csv_encoding: str | None = None
    output_csv_encoding: str = "utf-8-sig"

    # parallel
    workers: int = 60
    shard_count: int = 256
    submit_window: int = 12

    # fixed threshold
    fixed_time_quantile: float = 0.70
    fixed_time_upper_scale: float = 1.5

    # adaptive threshold
    kkod_std_threshold: float = 100.0
    kkod_min_samples: int = 50
    iforest_n_estimators: int = 15
    iforest_random_state: int = 2023

    # timesplit
    high_hours: tuple[int, ...] = (7, 8, 17, 18)
    flat_hours: tuple[int, ...] = (9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21)
    low_hours: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 22, 23)

    # legacy rule
    min_trip_duration_sec: int = 300


# ============================ explicit config section ============================
CONFIG = PartitionConfig(
    excluded_input_path=Path(r"./data/center_excluded_supplement_df.parquet"),
    bayonet_match_path=Path(r"./data/center_realpos.csv"),
    output_dir=Path(r"./output/trajectory_partition_excluded"),
)
# ================================================================================


def config_to_jsonable(cfg: PartitionConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(PartitionConfig):
        value = getattr(cfg, f.name)
        if isinstance(value, Path):
            out[f.name] = str(value)
        elif isinstance(value, tuple):
            out[f.name] = list(value)
        else:
            out[f.name] = value
    return out


def build_config_from_mapping(base_cfg: PartitionConfig, patch: dict[str, Any]) -> PartitionConfig:
    merged = config_to_jsonable(base_cfg)
    merged.update(patch or {})

    kwargs: dict[str, Any] = {}
    for f in fields(PartitionConfig):
        if f.name not in merged:
            raise ValueError(f"missing config field: {f.name}")
        default_value = getattr(base_cfg, f.name)
        raw = merged[f.name]
        if isinstance(default_value, Path):
            kwargs[f.name] = Path(raw)
        elif isinstance(default_value, tuple):
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"{f.name} expects list/tuple")
            kwargs[f.name] = tuple(raw)
        else:
            kwargs[f.name] = raw
    return PartitionConfig(**kwargs)


def load_config_from_json(path: Path, base_cfg: PartitionConfig) -> PartitionConfig:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config json root must be object")
    cfg = build_config_from_mapping(base_cfg, payload)
    print(f"[config] loaded: {path}")
    return cfg


def dump_config_template(path: Path, base_cfg: PartitionConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config_to_jsonable(base_cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[config] template dumped: {path}")


def read_json_object(path: Path, where: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{where} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{where} root must be object: {path}")
    return payload


def apply_bayonet_pipeline_config(base_cfg: PartitionConfig, bayonet_cfg_path: Path) -> PartitionConfig:
    payload = read_json_object(bayonet_cfg_path, "bayonet pipeline config")
    patch: dict[str, Any] = {}

    if "excluded_supplement_data_path" in payload:
        patch["excluded_input_path"] = payload["excluded_supplement_data_path"]
    if "route_matched_bayonet_csv_path" in payload:
        patch["bayonet_match_path"] = payload["route_matched_bayonet_csv_path"]
    if "parallel_workers" in payload:
        patch["workers"] = payload["parallel_workers"]
    if "parallel_submit_window" in payload:
        patch["submit_window"] = payload["parallel_submit_window"]
    if "raw_csv_encoding" in payload:
        patch["input_csv_encoding"] = payload["raw_csv_encoding"]

    # Align output location by default with excluded output directory.
    # Excluded path in upstream config may end with .csv while actual file is parquet.
    if "excluded_supplement_data_path" in payload:
        excluded_path = Path(str(payload["excluded_supplement_data_path"]))
        patch["output_dir"] = excluded_path.parent / "trajectory_partition_excluded"

    cfg = build_config_from_mapping(base_cfg, patch)
    print(f"[config] aligned from bayonet pipeline config: {bayonet_cfg_path}")
    return cfg


def resolve_parallel_workers(configured_workers: int) -> int:
    if configured_workers <= 0:
        cpu = os.cpu_count() or 1
        return max(1, cpu - 2)
    return max(1, configured_workers)


def assert_required_columns(df: pd.DataFrame, required_columns: Iterable[str], where: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{where} missing required columns: {missing}")


def read_csv_with_fallback(path: Path, preferred_encoding: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"csv file not found: {path}")

    encodings = [preferred_encoding] if preferred_encoding else []
    encodings += ["gbk", "gb18030", "utf-8-sig", "utf-8"]
    last_error: Exception | None = None
    for enc in [x for x in encodings if x]:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"[read] {path.name} encoding={enc}")
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"cannot read csv: {path}, last_error={last_error}")


def read_table_preferring_parquet(path: Path, csv_encoding: str | None) -> tuple[pd.DataFrame, Path]:
    parquet_path = path if path.suffix.lower() == ".parquet" else path.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path), parquet_path
    if path.exists():
        return read_csv_with_fallback(path, preferred_encoding=csv_encoding), path
    raise FileNotFoundError(f"table not found: parquet={parquet_path}, csv={path}")


def load_bayonet_to_pos_map(path: Path, csv_encoding: str | None) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"bayonet match file not found: {path}")

    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
        if not isinstance(arr, np.ndarray) or len(arr) < 2:
            raise ValueError(f"unexpected npy format: {path}")
        node_list = list(arr[0])
        bayonet_list = list(arr[1])
        mapping = {
            str(kkbh).strip(): str(node).strip()
            for node, kkbh in zip(node_list, bayonet_list)
            if str(kkbh).strip() and str(node).strip()
        }
        print(f"[input] bayonet map size={len(mapping)}")
        return mapping

    df = read_csv_with_fallback(path, preferred_encoding=csv_encoding)
    assert_required_columns(df, ["KKBH"], "bayonet match csv")
    node_candidates = ["node", "pos_name", "NODE", "realpos"]
    node_col = next((col for col in node_candidates if col in df.columns), None)
    if node_col is None:
        raise ValueError(f"cannot find node column in {path}")

    mapping = {
        str(kkbh).strip(): str(node).strip()
        for kkbh, node in zip(df["KKBH"], df[node_col])
        if str(kkbh).strip() and str(node).strip()
    }
    print(f"[input] bayonet map size={len(mapping)}")
    return mapping


def normalize_excluded_records(data: pd.DataFrame) -> pd.DataFrame:
    # Latest bayonet_data_pipeline excluded output keeps many extra columns.
    # Trajectory partition only depends on the fields below.
    required = ["GCRQ", "GCSJ", "KKBH", "CLTMBH"]
    assert_required_columns(data, required, "excluded input")

    x = data.copy()
    x["CLTMBH"] = x["CLTMBH"].astype(str).str.strip()
    x["KKBH"] = x["KKBH"].astype(str).str.strip()
    x["GCRQ"] = x["GCRQ"].astype(str).str.split(" ", n=1, expand=True)[0]
    x["GCSJ"] = pd.to_datetime(x["GCSJ"], errors="coerce")

    x = x.dropna(subset=["GCRQ", "GCSJ", "KKBH", "CLTMBH"]).copy()
    x = x.drop_duplicates(subset=["GCRQ", "GCSJ", "CLTMBH"], keep="first")
    x = x.sort_values(by=["CLTMBH", "GCRQ", "GCSJ"]).reset_index(drop=True)
    print("[schema] using excluded columns: GCRQ,GCSJ,KKBH,CLTMBH")
    return x


def build_adjacent_travel_time(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values(by=["CLTMBH", "GCRQ", "GCSJ"]).reset_index(drop=True)

    if len(ordered) <= 1:
        return pd.DataFrame(columns=["KKOD", "time", "hour"])

    next_plate = ordered["CLTMBH"].shift(-1)
    next_date = ordered["GCRQ"].shift(-1)
    next_kkbh = ordered["KKBH"].shift(-1)
    next_time = ordered["GCSJ"].shift(-1)

    same_plate = ordered["CLTMBH"].eq(next_plate)
    same_date = ordered["GCRQ"].eq(next_date)
    valid = same_plate & same_date

    out = pd.DataFrame(
        {
            "KKOD": ordered["KKBH"].astype(str) + "-" + next_kkbh.astype(str),
            "time": (next_time - ordered["GCSJ"]).dt.total_seconds(),
            "hour": ordered["GCSJ"].dt.hour,
        }
    )
    out = out[valid].copy()
    out = out.dropna(subset=["KKOD", "time", "hour"])
    out = out[out["time"] >= 0].reset_index(drop=True)
    return out


def _iforest_filter(values: np.ndarray, cfg: PartitionConfig) -> np.ndarray:
    if values.size <= 2:
        return values

    model = IsolationForest(
        n_estimators=cfg.iforest_n_estimators,
        max_samples="auto",
        contamination="auto",
        random_state=cfg.iforest_random_state,
        max_features=1,
    )
    xx = values.reshape(-1, 1)
    model.fit(xx)
    labels = model.predict(xx)
    kept = values[labels == 1]
    return kept if kept.size > 0 else values


def build_dynamic_thresholds(travel_time_df: pd.DataFrame, cfg: PartitionConfig) -> pd.DataFrame:
    # Keep logic aligned with notebook:
    # 1) std<100 or sample<50 => period=all
    # 2) if any period is empty => period=all
    # 3) otherwise build IF per period and keep anomaly==1
    rows: list[list[Any]] = []

    timesplit = [set(cfg.high_hours), set(cfg.flat_hours), set(cfg.low_hours)]
    kkod_name = list(travel_time_df["KKOD"].drop_duplicates())
    print(f"[threshold] KKOD groups={len(kkod_name)}")

    start = time.perf_counter()
    for idx, kkod in enumerate(kkod_name, start=1):
        test = travel_time_df[travel_time_df["KKOD"] == kkod]
        if test.empty:
            continue

        if test["time"].std() < cfg.kkod_std_threshold or len(test) < cfg.kkod_min_samples:
            rows.append(
                [
                    kkod,
                    "all",
                    float(np.min(test["time"])),
                    float(np.median(test["time"])),
                    float(np.max(test["time"])),
                ]
            )
        else:
            degraded_to_all = False
            for j in range(0, 3):
                test_hour = test[test["hour"].isin(timesplit[j])].copy()

                if j == 0:
                    test_hour1 = test[test["hour"].isin(timesplit[j])]
                    test_hour2 = test[test["hour"].isin(timesplit[j + 1])]
                    test_hour3 = test[test["hour"].isin(timesplit[j + 2])]
                    if test_hour1.empty or test_hour2.empty or test_hour3.empty:
                        rows.append(
                            [
                                kkod,
                                "all",
                                float(np.min(test["time"])),
                                float(np.median(test["time"])),
                                float(np.max(test["time"])),
                            ]
                        )
                        degraded_to_all = True
                        break

                if j == 0:
                    label = "high"
                elif j == 1:
                    label = "flat"
                else:
                    label = "low"

                values = test_hour["time"].to_numpy(dtype=float)
                filtered = _iforest_filter(values, cfg)
                rows.append(
                    [
                        kkod,
                        label,
                        float(np.min(filtered)),
                        float(np.median(filtered)),
                        float(np.max(filtered)),
                    ]
                )

        if idx % 5000 == 0:
            elapsed = max(time.perf_counter() - start, 1e-9)
            print(f"[threshold] processed={idx}/{len(kkod_name)}, groups_per_sec={idx / elapsed:.2f}")

    return pd.DataFrame(rows, columns=["kkod", "period", "min", "median", "max"])


def build_kkod_period_map(threshold_df: pd.DataFrame) -> dict[str, dict[str, tuple[Any, Any]]]:
    out: dict[str, dict[str, tuple[Any, Any]]] = {}
    for row in threshold_df.itertuples(index=False):
        kkod = str(row.kkod)
        period = str(row.period)
        out.setdefault(kkod, {})[period] = (row.min, row.max)
    return out


def _is_nan_like(v: Any) -> bool:
    if pd.isna(v):
        return True
    return str(v).strip().lower() == "nan"


def _resolve_period(hour: int, cfg: PartitionConfig) -> str:
    if hour in set(cfg.high_hours):
        return "high"
    if hour in set(cfg.flat_hours):
        return "flat"
    return "low"


def resolve_mintime_maxtime_legacy(
    kkod: str,
    hour: int,
    kkod_map: dict[str, dict[str, tuple[Any, Any]]],
    fixed_b: float,
    cfg: PartitionConfig,
) -> tuple[int, int]:
    # Notebook default fallback
    default_min = 1
    default_max = round(cfg.fixed_time_upper_scale * fixed_b)

    test_range = kkod_map.get(kkod)
    if test_range is None:
        return default_min, default_max

    if len(test_range) == 1:
        # Equivalent to "all" in notebook
        min_raw, max_raw = next(iter(test_range.values()))
        if _is_nan_like(min_raw) or _is_nan_like(max_raw):
            mintime = default_min
            maxtime = default_max
        else:
            kkod_mintime = float(min_raw)
            kkod_maxtime = float(max_raw)
            kkod_interval = kkod_maxtime - kkod_mintime
            if kkod_mintime >= fixed_b and kkod_interval < 60:
                mintime = default_min
                maxtime = default_max
            else:
                mintime = int(kkod_mintime)
                maxtime = round(kkod_maxtime)
    else:
        period = _resolve_period(hour, cfg)
        if period not in test_range:
            # Notebook data should normally contain this row; fallback to default keeps run stable.
            mintime = default_min
            maxtime = default_max
        else:
            min_raw, max_raw = test_range[period]
            if _is_nan_like(min_raw) or _is_nan_like(max_raw):
                mintime = default_min
                maxtime = default_max
            else:
                kkod_mintime = float(min_raw)
                kkod_maxtime = float(max_raw)
                kkod_interval = kkod_maxtime - kkod_mintime
                if kkod_mintime >= fixed_b and kkod_interval < 60:
                    mintime = default_min
                    maxtime = default_max
                else:
                    mintime = int(kkod_mintime)
                    maxtime = round(kkod_maxtime)

    maxtime = min(maxtime, round(cfg.fixed_time_upper_scale * fixed_b))
    return mintime, maxtime


def _append_trip_if_valid(
    st_pos: list[str],
    st_bayonet: list[str],
    st_time: list[pd.Timestamp],
    cltmbh: str,
    out: dict[str, list[Any]],
    min_trip_duration_sec: int,
) -> None:
    if len(st_pos) <= 1:
        return
    duration = (st_time[-1] - st_time[0]).total_seconds()
    if duration <= min_trip_duration_sec:
        return
    out["path_pos"].append(list(st_pos))
    out["path_bayonet"].append(list(st_bayonet))
    out["path_time"].append(list(st_time))
    out["path_start_time"].append(st_time[0])
    out["path_stop_time"].append(st_time[-1])
    out["path_travel"].append(duration)
    out["path_cltmbh"].append(cltmbh)


def process_vehicle_legacy(
    vt: pd.DataFrame,
    kkod_map: dict[str, dict[str, tuple[Any, Any]]],
    bayonet_to_pos: dict[str, str],
    fixed_b: float,
    cfg: PartitionConfig,
) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {
        "path_pos": [],
        "path_bayonet": [],
        "path_time": [],
        "path_start_time": [],
        "path_stop_time": [],
        "path_travel": [],
        "path_cltmbh": [],
    }

    if len(vt) <= 1:
        return out

    cltmbh = str(vt.iloc[0]["CLTMBH"])

    vt_pos: list[str] = []
    vt_bayonet: list[str] = []
    vt_time: list[pd.Timestamp] = []

    for row in vt.itertuples(index=False):
        kkbh = str(getattr(row, "KKBH"))
        pos = bayonet_to_pos.get(kkbh)
        if pos is None:
            # Keep behavior stable for mapping gaps: skip unmatched point.
            continue
        vt_pos.append(str(pos))
        vt_bayonet.append(kkbh)
        vt_time.append(pd.Timestamp(getattr(row, "GCSJ")))

    if len(vt_pos) <= 1:
        return out

    st_pos: list[str] = []
    st_bayonet: list[str] = []
    st_time: list[pd.Timestamp] = []

    for z in range(1, len(vt_pos)):
        kkod = f"{vt_bayonet[z - 1]}-{vt_bayonet[z]}"
        kkhour = vt_time[z].hour
        mintime, maxtime = resolve_mintime_maxtime_legacy(kkod, kkhour, kkod_map, fixed_b, cfg)

        if st_time == []:
            last_time = vt_time[z - 1]
        else:
            last_time = st_time[-1]
        ti = (vt_time[z] - last_time).total_seconds()

        if vt_bayonet[z] != vt_bayonet[z - 1]:
            if ti >= mintime and ti <= maxtime and ti != 0:
                if z == 1:
                    st_pos.extend((vt_pos[z - 1], vt_pos[z]))
                    st_bayonet.extend((vt_bayonet[z - 1], vt_bayonet[z]))
                    st_time.extend((vt_time[z - 1], vt_time[z]))
                elif z == (len(vt_pos) - 1):
                    st_pos.append(vt_pos[z])
                    st_bayonet.append(vt_bayonet[z])
                    st_time.append(vt_time[z])
                    _append_trip_if_valid(
                        st_pos,
                        st_bayonet,
                        st_time,
                        cltmbh,
                        out,
                        cfg.min_trip_duration_sec,
                    )
                else:
                    st_pos.append(vt_pos[z])
                    st_bayonet.append(vt_bayonet[z])
                    st_time.append(vt_time[z])
            elif ti > maxtime:
                _append_trip_if_valid(
                    st_pos,
                    st_bayonet,
                    st_time,
                    cltmbh,
                    out,
                    cfg.min_trip_duration_sec,
                )
                st_pos = [vt_pos[z]]
                st_bayonet = [vt_bayonet[z]]
                st_time = [vt_time[z]]
            else:
                continue
        elif ti > fixed_b:
            _append_trip_if_valid(
                st_pos,
                st_bayonet,
                st_time,
                cltmbh,
                out,
                cfg.min_trip_duration_sec,
            )
            st_pos = [vt_pos[z]]
            st_bayonet = [vt_bayonet[z]]
            st_time = [vt_time[z]]
        else:
            continue

    return out


def _merge_result(dst: dict[str, list[Any]], src: dict[str, list[Any]]) -> None:
    for k in dst.keys():
        dst[k].extend(src[k])


_WORKER_KKOD_MAP: dict[str, dict[str, tuple[Any, Any]]] | None = None
_WORKER_BAYONET_MAP: dict[str, str] | None = None
_WORKER_B: float = 0.0
_WORKER_CFG: PartitionConfig | None = None


def _init_worker(
    kkod_map: dict[str, dict[str, tuple[Any, Any]]],
    bayonet_to_pos: dict[str, str],
    fixed_b: float,
    cfg: PartitionConfig,
) -> None:
    global _WORKER_KKOD_MAP, _WORKER_BAYONET_MAP, _WORKER_B, _WORKER_CFG
    _WORKER_KKOD_MAP = kkod_map
    _WORKER_BAYONET_MAP = bayonet_to_pos
    _WORKER_B = fixed_b
    _WORKER_CFG = cfg


def _process_shard_file(shard_path: str) -> dict[str, list[Any]]:
    if _WORKER_KKOD_MAP is None or _WORKER_BAYONET_MAP is None or _WORKER_CFG is None:
        raise RuntimeError("worker context is not initialized")

    out: dict[str, list[Any]] = {
        "path_pos": [],
        "path_bayonet": [],
        "path_time": [],
        "path_start_time": [],
        "path_stop_time": [],
        "path_travel": [],
        "path_cltmbh": [],
    }

    df = pd.read_parquet(shard_path)
    if df.empty:
        return out
    df = df.sort_values(by=["CLTMBH", "GCSJ"]).reset_index(drop=True)

    for vid, vt in df.groupby("CLTMBH", sort=False):
        vt = vt.reset_index(drop=True)
        seg = process_vehicle_legacy(
            vt=vt,
            kkod_map=_WORKER_KKOD_MAP,
            bayonet_to_pos=_WORKER_BAYONET_MAP,
            fixed_b=_WORKER_B,
            cfg=_WORKER_CFG,
        )
        _merge_result(out, seg)
    return out


def _create_partition_shards(records: pd.DataFrame, shard_count: int, shard_dir: Path) -> list[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_count = max(1, shard_count)

    x = records.copy()
    hashed = pd.util.hash_pandas_object(x["CLTMBH"], index=False).to_numpy(dtype=np.uint64)
    x["__shard"] = (hashed % np.uint64(shard_count)).astype(np.int64)

    paths: list[Path] = []
    for shard_id, grp in x.groupby("__shard", sort=True):
        out_path = shard_dir / f"shard_{int(shard_id):04d}.parquet"
        grp[["CLTMBH", "KKBH", "GCSJ"]].to_parquet(out_path, index=False)
        paths.append(out_path)
    return paths


def run_partition(
    records: pd.DataFrame,
    kkod_map: dict[str, dict[str, tuple[Any, Any]]],
    bayonet_to_pos: dict[str, str],
    fixed_b: float,
    cfg: PartitionConfig,
) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {
        "path_pos": [],
        "path_bayonet": [],
        "path_time": [],
        "path_start_time": [],
        "path_stop_time": [],
        "path_travel": [],
        "path_cltmbh": [],
    }

    vehicle_ids = list(records["CLTMBH"].drop_duplicates())
    workers = resolve_parallel_workers(cfg.workers)
    run_tmp_dir = cfg.output_dir / "_traj_partition_tmp" / f"run_{int(time.time())}_{os.getpid()}"
    shard_dir = run_tmp_dir / "shards"
    shard_paths = _create_partition_shards(records, cfg.shard_count, shard_dir)
    print(f"[partition] vehicles={len(vehicle_ids)}, shards={len(shard_paths)}, workers={workers}")
    start = time.perf_counter()

    try:
        if workers == 1 or len(shard_paths) == 1:
            _init_worker(kkod_map, bayonet_to_pos, fixed_b, cfg)
            for i, path in enumerate(shard_paths, start=1):
                part = _process_shard_file(str(path))
                _merge_result(result, part)
                if i % max(1, cfg.submit_window) == 0:
                    elapsed = max(time.perf_counter() - start, 1e-9)
                    print(f"[partition] done_shards={i}/{len(shard_paths)}, shards_per_sec={i / elapsed:.2f}")
            return result

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(kkod_map, bayonet_to_pos, fixed_b, cfg),
        ) as pool:
            pending: dict[Any, int] = {}
            next_idx = 0
            done_count = 0

            while next_idx < len(shard_paths) and len(pending) < max(1, cfg.submit_window):
                fut = pool.submit(_process_shard_file, str(shard_paths[next_idx]))
                pending[fut] = next_idx
                next_idx += 1

            while pending:
                done = next(as_completed(list(pending.keys())))
                done_count += 1
                part = done.result()
                _merge_result(result, part)
                pending.pop(done)

                if next_idx < len(shard_paths):
                    fut = pool.submit(_process_shard_file, str(shard_paths[next_idx]))
                    pending[fut] = next_idx
                    next_idx += 1

                if done_count % max(1, cfg.submit_window) == 0:
                    elapsed = max(time.perf_counter() - start, 1e-9)
                    print(
                        f"[partition] done_shards={done_count}/{len(shard_paths)}, "
                        f"shards_per_sec={done_count / elapsed:.2f}"
                    )
        return result
    finally:
        shutil.rmtree(run_tmp_dir, ignore_errors=True)


def save_outputs(
    travel_time_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    fixed_b: float,
    result: dict[str, list[Any]],
    cfg: PartitionConfig,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.output_prefix

    # threshold artifacts
    travel_csv = cfg.output_dir / f"{prefix}all_adjacent_travel_time.csv"
    threshold_csv = cfg.output_dir / f"{prefix}kkod_time_threshold.csv"
    threshold_npy = cfg.output_dir / f"{prefix}kkod_time_threshold.npy"
    meta_json = cfg.output_dir / f"{prefix}partition_meta.json"

    travel_time_df.to_csv(travel_csv, index=False, encoding=cfg.output_csv_encoding)
    threshold_df.to_csv(threshold_csv, index=False, encoding=cfg.output_csv_encoding)
    np.save(threshold_npy, threshold_df[["kkod", "period", "min", "median", "max"]].to_numpy(dtype=object))

    meta = {
        "fixed_time_quantile": cfg.fixed_time_quantile,
        "fixed_time_threshold_b": fixed_b,
        "fixed_time_upper_scale": cfg.fixed_time_upper_scale,
        "travel_time_rows": int(len(travel_time_df)),
        "kkod_threshold_rows": int(len(threshold_df)),
        "segmented_trip_count": int(len(result["path_pos"])),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # path outputs
    np.save(cfg.output_dir / f"{prefix}path_pos.npy", np.array(result["path_pos"], dtype=object))
    np.save(cfg.output_dir / f"{prefix}path_bayonet.npy", np.array(result["path_bayonet"], dtype=object))
    np.save(cfg.output_dir / f"{prefix}path_time.npy", np.array(result["path_time"], dtype=object))
    np.save(cfg.output_dir / f"{prefix}path_start_time.npy", np.array(result["path_start_time"], dtype=object))
    np.save(cfg.output_dir / f"{prefix}path_stop_time.npy", np.array(result["path_stop_time"], dtype=object))
    np.save(cfg.output_dir / f"{prefix}path_travel.npy", np.array(result["path_travel"], dtype=object))
    np.save(cfg.output_dir / f"{prefix}path_cltmbh.npy", np.array(result["path_cltmbh"], dtype=object))

    bayonet_number = [len(x) for x in result["path_bayonet"]]
    np.save(cfg.output_dir / f"{prefix}path_bayonet_number.npy", np.array(bayonet_number, dtype=object))

    print(f"[output] saved in: {cfg.output_dir}")
    print(f"[output] segmented trips: {len(result['path_pos'])}")


def run_pipeline(cfg: PartitionConfig) -> None:
    t0 = time.perf_counter()
    print("\n========== trajectory partition ==========")
    print(f"excluded_input_path: {cfg.excluded_input_path}")
    print(f"bayonet_match_path: {cfg.bayonet_match_path}")
    print(f"output_dir: {cfg.output_dir}")
    print(f"workers: {resolve_parallel_workers(cfg.workers)}")
    print(f"shard_count: {cfg.shard_count}")
    print("scope: excluded_only")
    print("semantics: legacy_notebook")
    print("==========================================\n")

    excluded_raw, source_path = read_table_preferring_parquet(cfg.excluded_input_path, cfg.input_csv_encoding)
    print(f"[input] loaded: {source_path}, rows={len(excluded_raw)}")
    excluded = normalize_excluded_records(excluded_raw)
    print(f"[input] normalized rows={len(excluded)}")
    if excluded.empty:
        raise ValueError("excluded input is empty after normalization")

    travel_time_df = build_adjacent_travel_time(excluded)
    if travel_time_df.empty:
        raise ValueError("adjacent travel-time table is empty")
    print(f"[threshold] adjacent rows={len(travel_time_df)}")

    fixed_b = float(travel_time_df["time"].quantile(cfg.fixed_time_quantile))
    print(f"[threshold] B={fixed_b:.4f}s")

    threshold_df = build_dynamic_thresholds(travel_time_df, cfg)
    if threshold_df.empty:
        raise ValueError("dynamic threshold table is empty")
    kkod_map = build_kkod_period_map(threshold_df)
    print(f"[threshold] threshold rows={len(threshold_df)}")

    bayonet_to_pos = load_bayonet_to_pos_map(cfg.bayonet_match_path, cfg.input_csv_encoding)

    # Keep only route-mapped records for partition
    before = len(excluded)
    excluded = excluded[excluded["KKBH"].astype(str).isin(set(bayonet_to_pos.keys()))].copy()
    excluded = excluded.sort_values(by=["CLTMBH", "GCSJ", "GCRQ"]).reset_index(drop=True)
    print(f"[partition] route-mapped rows={len(excluded)}/{before}")
    if excluded.empty:
        raise ValueError("no route-mapped excluded records left")

    result = run_partition(
        records=excluded[["CLTMBH", "KKBH", "GCSJ"]],
        kkod_map=kkod_map,
        bayonet_to_pos=bayonet_to_pos,
        fixed_b=fixed_b,
        cfg=cfg,
    )

    save_outputs(travel_time_df, threshold_df, fixed_b, result, cfg)
    elapsed = time.perf_counter() - t0
    print(f"\nfinished, elapsed={elapsed:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="trajectory partition pipeline (excluded-only, legacy semantics)")
    parser.add_argument(
        "--bayonet-config",
        type=str,
        help="bayonet_data_pipeline json config path; auto-align excluded input and bayonet map paths",
    )
    parser.add_argument("--config", type=str, help="external JSON config path")
    parser.add_argument("--dump-config-template", type=str, help="dump JSON config template and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runtime_cfg = CONFIG
    if args.dump_config_template:
        dump_config_template(Path(args.dump_config_template), CONFIG)
        raise SystemExit(0)
    if args.bayonet_config:
        runtime_cfg = apply_bayonet_pipeline_config(runtime_cfg, Path(args.bayonet_config))
    if args.config:
        runtime_cfg = load_config_from_json(Path(args.config), runtime_cfg)
    run_pipeline(runtime_cfg)
