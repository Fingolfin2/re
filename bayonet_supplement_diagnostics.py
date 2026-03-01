#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
被排除样本补算诊断脚本。

用途：
1. 核查切分前清洗损失（dropna vs drop_duplicates）。
2. 核查去重键 (GCRQ, GCSJ, CLTMBH) 的冲突风险。
3. 核查 excluded 中未映射到路网卡口的 KKBH 分布。
4. 对比补算后 CLLX/RLLX 分布（与车辆表、与纳入样本、与原始 CSV 分布）。
5. 统计最终结果（included_mapped + excluded_supplement）中的本地车牌占比。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from bayonet_data_pipeline import (
    CONFIG,
    PipelineConfig,
    assert_required_columns,
    build_raw_sampling_distribution,
    filter_center_area_records,
    load_bayonet_base,
    load_config_from_json,
    load_vehicle_table_for_attr_imputation,
    print_config_summary,
    read_csv_with_fallback,
    read_parquet_with_hint,
    resolve_parquet_path,
    split_included_and_excluded_records,
)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _fmt_ratio(numerator: int, denominator: int) -> str:
    return f"{_ratio(numerator, denominator) * 100:.2f}%"


def _read_table_preferring_parquet(path: Path, where: str, csv_encoding: str) -> pd.DataFrame:
    parquet_path = resolve_parquet_path(path)
    if parquet_path.exists():
        return read_parquet_with_hint(parquet_path, where)
    if path.exists():
        return pd.read_csv(path, encoding=csv_encoding, low_memory=False)
    raise FileNotFoundError(f"{where} 文件不存在: parquet={parquet_path}, csv={path}")


def _to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _build_pair_distribution(df: pd.DataFrame, cllx_col: str = "CLLX", rllx_col: str = "RLLX") -> pd.DataFrame:
    assert_required_columns(df, [cllx_col, rllx_col], "分布统计输入数据")
    x = df[[cllx_col, rllx_col]].copy()
    x[cllx_col] = x[cllx_col].astype(str).str.strip().str[:2]
    x[rllx_col] = x[rllx_col].astype(str).str.strip()
    x = x[
        x[cllx_col].ne("")
        & x[rllx_col].ne("")
        & x[cllx_col].str.lower().ne("nan")
        & x[rllx_col].str.lower().ne("nan")
    ]
    if x.empty:
        return pd.DataFrame(columns=["CLLX", "RLLX", "count", "ratio"])

    out = (
        x.value_counts([cllx_col, rllx_col])
        .rename("count")
        .reset_index()
        .rename(columns={cllx_col: "CLLX", rllx_col: "RLLX"})
    )
    out["ratio"] = out["count"] / out["count"].sum()
    return out.sort_values("count", ascending=False, ignore_index=True)


def _compare_distributions(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    l = left[["CLLX", "RLLX", "count", "ratio"]].rename(
        columns={"count": f"{left_name}_count", "ratio": f"{left_name}_ratio"}
    )
    r = right[["CLLX", "RLLX", "count", "ratio"]].rename(
        columns={"count": f"{right_name}_count", "ratio": f"{right_name}_ratio"}
    )
    merged = l.merge(r, on=["CLLX", "RLLX"], how="outer").fillna(0)
    merged["delta_pp"] = (merged[f"{left_name}_ratio"] - merged[f"{right_name}_ratio"]) * 100.0
    merged["abs_delta_pp"] = merged["delta_pp"].abs()
    return merged.sort_values("abs_delta_pp", ascending=False, ignore_index=True)


def _save_csv(df: pd.DataFrame, output_dir: Path | None, filename: str) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[导出] {out_path}")


def _normalize_for_split_with_profile(data: pd.DataFrame, topk: int) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """
    与主脚本 normalize_records_for_split 保持同口径，但返回清洗画像信息。
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

    before_dropna = len(x)
    x_dropna = x.dropna(subset=["GCRQ", "GCSJ", "CLTMBH", "KKBH"]).copy()
    after_dropna = len(x_dropna)
    dropna_removed = before_dropna - after_dropna

    dedup_keys = ["GCRQ", "GCSJ", "CLTMBH"]
    dup_all = x_dropna[x_dropna.duplicated(subset=dedup_keys, keep=False)].copy()
    x_dedup = x_dropna.drop_duplicates(subset=dedup_keys, keep="first").copy()
    after_dedup = len(x_dedup)
    dedup_removed = after_dropna - after_dedup

    conflict_top = pd.DataFrame()
    duplicate_group_count = 0
    conflict_group_count = 0
    conflict_row_count = 0

    if not dup_all.empty:
        grp = dup_all.groupby(dedup_keys, sort=False, dropna=False)
        base = grp.size().rename("group_size").to_frame()
        for col in ["KKBH", "FZJG", "CLLX", "RLLX"]:
            base[f"{col}_nunique"] = grp[col].nunique(dropna=False)

        base["has_conflict"] = (
            (base["KKBH_nunique"] > 1)
            | (base["FZJG_nunique"] > 1)
            | (base["CLLX_nunique"] > 1)
            | (base["RLLX_nunique"] > 1)
        )
        duplicate_group_count = len(base)
        conflict = base[base["has_conflict"]].copy()
        conflict_group_count = len(conflict)
        conflict_row_count = int(conflict["group_size"].sum()) if not conflict.empty else 0

        conflict_top = (
            conflict.sort_values("group_size", ascending=False)
            .reset_index()
            .head(max(1, int(topk)))
        )

    x_norm = x_dedup.sort_values(by=["GCRQ", "GCSJ", "CLTMBH"]).reset_index(drop=True)
    stats = {
        "before_dropna": before_dropna,
        "after_dropna": after_dropna,
        "dropna_removed": dropna_removed,
        "after_dedup": after_dedup,
        "dedup_removed": dedup_removed,
        "total_removed": before_dropna - after_dedup,
        "duplicate_group_count": duplicate_group_count,
        "conflict_group_count": conflict_group_count,
        "conflict_row_count": conflict_row_count,
    }
    return x_norm, stats, conflict_top


def _print_distribution(name: str, dist: pd.DataFrame, topk: int) -> None:
    print(f"[分布] {name}（Top{topk}）:")
    if dist.empty:
        print("  (空)")
        return
    show = dist.head(topk).copy()
    for _, row in show.iterrows():
        print(f"  {row['CLLX']}/{row['RLLX']}: count={int(row['count'])}, ratio={row['ratio'] * 100:.2f}%")


def run_diagnostics(
    config: PipelineConfig,
    topk: int = 20,
    output_dir: Path | None = None,
    skip_raw_sampling_dist: bool = False,
) -> None:
    print("\n========== 补算诊断统计 ==========")
    print_config_summary(config)

    bayonets = load_bayonet_base(config)
    bayonet_ids = set(bayonets["KKBH"].astype(str).str.strip())
    center_df = filter_center_area_records(config, bayonet_ids)
    print(f"[诊断] 中心城区记录（切分前）: {len(center_df)}")

    normalized_df, clean_stats, conflict_top = _normalize_for_split_with_profile(center_df, topk=topk)
    del center_df
    print(
        f"[诊断] 切分前清洗移除: {clean_stats['total_removed']} "
        f"(dropna={clean_stats['dropna_removed']}, dedup={clean_stats['dedup_removed']})"
    )
    print(
        "[诊断] 去重键冲突组: "
        f"{clean_stats['conflict_group_count']}/{clean_stats['duplicate_group_count']} 组，"
        f"冲突行数={clean_stats['conflict_row_count']}"
    )

    if not conflict_top.empty:
        _save_csv(conflict_top, output_dir, "dedup_key_conflict_groups_top.csv")

    included_df, excluded_df = split_included_and_excluded_records(normalized_df, config)
    del normalized_df

    route_df = read_csv_with_fallback(config.route_matched_bayonet_csv_path)
    assert_required_columns(route_df, ["KKBH"], "路网映射卡口文件")
    route_set = set(route_df["KKBH"].astype(str).str.strip())
    del route_df

    included_mask = included_df["KKBH"].astype(str).str.strip().isin(route_set)
    excluded_mask = excluded_df["KKBH"].astype(str).str.strip().isin(route_set)
    included_mapped = included_df.loc[included_mask].copy()
    excluded_mapped = excluded_df.loc[excluded_mask].copy()
    excluded_unmapped = excluded_df.loc[~excluded_mask].copy()

    print(
        "[诊断] excluded 路网映射: "
        f"mapped={len(excluded_mapped)} ({_fmt_ratio(len(excluded_mapped), len(excluded_df))}), "
        f"unmapped={len(excluded_unmapped)} ({_fmt_ratio(len(excluded_unmapped), len(excluded_df))})"
    )

    unmapped_top = (
        excluded_unmapped["KKBH"]
        .astype(str)
        .str.strip()
        .value_counts()
        .rename_axis("KKBH")
        .reset_index(name="count")
        .head(topk)
    )
    print(f"[诊断] 未映射 KKBH Top{topk} 已生成。")
    _save_csv(unmapped_top, output_dir, "excluded_unmapped_kkbh_top.csv")

    route_set_no_zero = {k.lstrip("0") for k in route_set if k}
    normalized_hit = (
        excluded_unmapped["KKBH"].astype(str).str.strip().str.lstrip("0").isin(route_set_no_zero).sum()
        if not excluded_unmapped.empty
        else 0
    )
    print(
        "[诊断] 未映射编码一致性线索(去前导0后可命中): "
        f"{normalized_hit}/{len(excluded_unmapped)} ({_fmt_ratio(int(normalized_hit), len(excluded_unmapped))})"
    )

    included_dist = _build_pair_distribution(included_df, "CLLX", "RLLX")
    _print_distribution("纳入样本(included)", included_dist, topk=min(topk, 10))

    vehicle_pool = load_vehicle_table_for_attr_imputation(config)
    vehicle_dist = _build_pair_distribution(vehicle_pool, "CLLX", "RLLX")
    _print_distribution("车辆信息表(设备级)", vehicle_dist, topk=min(topk, 10))

    supplement_df = _read_table_preferring_parquet(
        config.excluded_supplement_data_path,
        "被排除样本补算结果",
        config.output_csv_encoding,
    )
    supplement_dist = _build_pair_distribution(supplement_df, "CLLX", "RLLX")
    _print_distribution("补算结果(excluded_supplement)", supplement_dist, topk=min(topk, 10))

    device_attr_df = _read_table_preferring_parquet(
        config.excluded_device_attr_path,
        "设备属性分配明细",
        config.output_csv_encoding,
    )
    assert_required_columns(device_attr_df, ["CLLX", "RLLX", "assigned_by_sampling"], "设备属性分配明细")
    sampled_device_df = device_attr_df[_to_bool_series(device_attr_df["assigned_by_sampling"])].copy()
    sampled_device_dist = _build_pair_distribution(sampled_device_df, "CLLX", "RLLX")
    _print_distribution("采样分配设备(设备级)", sampled_device_dist, topk=min(topk, 10))

    comp_sup_vs_included = _compare_distributions(supplement_dist, included_dist, "supplement", "included")
    comp_sup_vs_vehicle = _compare_distributions(supplement_dist, vehicle_dist, "supplement", "vehicle")
    _save_csv(comp_sup_vs_included, output_dir, "dist_compare_supplement_vs_included.csv")
    _save_csv(comp_sup_vs_vehicle, output_dir, "dist_compare_supplement_vs_vehicle.csv")

    if not skip_raw_sampling_dist:
        raw_dist = build_raw_sampling_distribution(config).rename(columns={"p": "ratio"})
        raw_dist = raw_dist[["CLLX", "RLLX", "count", "ratio"]].copy()
        _print_distribution("原始CSV去重分布", raw_dist, topk=min(topk, 10))
        comp_sup_vs_raw = _compare_distributions(supplement_dist, raw_dist, "supplement", "raw")
        _save_csv(comp_sup_vs_raw, output_dir, "dist_compare_supplement_vs_raw.csv")
    else:
        print("[诊断] 已跳过原始CSV采样分布统计（--skip-raw-sampling-dist）。")

    local_prefixes = set(config.local_plate_prefixes)
    assert_required_columns(included_mapped, ["FZJG", "CLTMBH"], "included_mapped")
    assert_required_columns(supplement_df, ["FZJG", "CLTMBH"], "supplement_df")

    final_df = pd.concat(
        [
            included_mapped[["CLTMBH", "FZJG"]].copy(),
            supplement_df[["CLTMBH", "FZJG"]].copy(),
        ],
        ignore_index=True,
    )
    final_df["FZJG"] = final_df["FZJG"].astype(str).str.strip()
    final_df["CLTMBH"] = final_df["CLTMBH"].astype(str).str.strip()

    final_local_record = int(final_df["FZJG"].isin(local_prefixes).sum())
    final_total_record = len(final_df)
    final_device = final_df.drop_duplicates(subset=["CLTMBH"], keep="first")
    final_local_device = int(final_device["FZJG"].isin(local_prefixes).sum())
    final_total_device = len(final_device)

    print(
        "[诊断] 最终结果本地车牌占比(记录级): "
        f"{final_local_record}/{final_total_record} ({_fmt_ratio(final_local_record, final_total_record)})"
    )
    print(
        "[诊断] 最终结果本地车牌占比(设备级): "
        f"{final_local_device}/{final_total_device} ({_fmt_ratio(final_local_device, final_total_device)})"
    )

    sampled_devices = int(_to_bool_series(device_attr_df["assigned_by_sampling"]).sum())
    print(
        "[诊断] 设备属性来源: "
        f"采样设备={sampled_devices}/{len(device_attr_df)} ({_fmt_ratio(sampled_devices, len(device_attr_df))})"
    )

    if output_dir is not None:
        summary = pd.DataFrame(
            [
                ("center_records_before_split", len(included_df) + len(excluded_df)),
                ("clean_removed_total", clean_stats["total_removed"]),
                ("clean_removed_dropna", clean_stats["dropna_removed"]),
                ("clean_removed_dedup", clean_stats["dedup_removed"]),
                ("included_records", len(included_df)),
                ("excluded_records", len(excluded_df)),
                ("excluded_mapped_records", len(excluded_mapped)),
                ("excluded_unmapped_records", len(excluded_unmapped)),
                ("final_local_record_ratio", _ratio(final_local_record, final_total_record)),
                ("final_local_device_ratio", _ratio(final_local_device, final_total_device)),
            ],
            columns=["metric", "value"],
        )
        _save_csv(summary, output_dir, "diagnostics_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补算阶段诊断统计脚本")
    parser.add_argument("--config", type=str, help="外部 JSON 配置路径（覆盖脚本内 CONFIG）")
    parser.add_argument("--topk", type=int, default=20, help="Top K 输出数量")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="可选：明细 CSV 输出目录（默认不落盘，只打印）",
    )
    parser.add_argument(
        "--skip-raw-sampling-dist",
        action="store_true",
        help="跳过原始 CSV 去重分布统计（减少耗时）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = CONFIG
    if args.config:
        cfg = load_config_from_json(Path(args.config), CONFIG)
    out_dir = Path(args.output_dir) if args.output_dir else None
    run_diagnostics(
        config=cfg,
        topk=max(1, int(args.topk)),
        output_dir=out_dir,
        skip_raw_sampling_dist=bool(args.skip_raw_sampling_dist),
    )
