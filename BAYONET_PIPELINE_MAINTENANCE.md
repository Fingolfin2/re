# bayonet_data_pipeline 维护说明

## 1. 文件与职责
- 主流程脚本：`re/bayonet_data_pipeline.py`
- 诊断脚本：`re/bayonet_supplement_diagnostics.py`
- 配置模板：`re/bayonet_data_pipeline.config.example.json`

主流程负责三阶段：
1. 预处理
2. 统计
3. 被排除样本补算

## 2. 当前实现的关键口径

### 2.1 逻辑不变约束
- 重构和性能优化必须保持原有计算口径不变。
- 若怀疑原逻辑有问题，先与用户确认，再改计算逻辑。

### 2.2 切分前清洗口径
函数：`normalize_records_for_split`
- 标准化字段：`CLTMBH/KKBH/FZJG/CLLX/RLLX`
- `CLLX` 截前两位
- `GCRQ` 保留日期
- `GCSJ` 解析为时间
- `dropna(subset=["GCRQ","GCSJ","CLTMBH","KKBH"])`
- `drop_duplicates(subset=["GCRQ","GCSJ","CLTMBH"], keep="first")`

### 2.3 补算口径
- 仅对 `excluded` 样本补算，不重复补 `included`。
- 补算前先做路网可映射过滤。
- 设备在车辆表中：直接用表内 `CLLX/RLLX`。
- 设备不在车辆表中：按采样分布随机分配。
- 当前默认采样分布口径为“真实还原原始总体分布”：
  - 基于原始 CSV 去重后联合分布，不限制于 `A/B` 或 `K/H`。
  - 因此分布中出现 `C/O/AC`、新能源或其他编码属于预期行为。

## 3. 采样分布构建（已升级）

## 3.1 目标
在不改变口径的前提下，充分利用多核与大内存，解决原先单进程瓶颈。

## 3.2 新机制
- 支持采样分布缓存（命中后跳过全量扫描）。
- 支持并行构建采样分布（分桶落盘 + 并行聚合）。
- 并行失败可自动回退串行，保证流程可用性。

## 3.3 并行构建流程
1. 分块读取原始 CSV，按既有口径清洗。
2. 计算去重键哈希并按桶号写入中间 Parquet 分桶。
3. 多进程并行聚合分桶。
4. 每个分桶按 `__row_id` 排序后执行 `drop_duplicates(..., keep="first")`。
5. 合并各桶 `(CLLX,RLLX)` 计数并计算概率 `p`。

说明：
- 通过全局 `__row_id` 保证 `keep="first"` 语义，与原逻辑一致。
- 监控日志会出现 `sampling-dist phase=partition` 与 `phase=aggregate`。

## 3.4 缓存机制
- 缓存文件：
  - `raw_sampling_distribution_cache.parquet`
  - `raw_sampling_distribution_cache.meta.json`
- 缓存签名包含：
  - 原始文件路径、大小、mtime
  - `raw_csv_chunksize`
  - `raw_csv_encoding`
  - 逻辑版本号
- 签名不匹配自动重建。

## 3.5 诊断阶段为何可能再次构建 raw 分布
- 诊断脚本中，raw 分布用于“补算结果 vs 原始CSV基准分布”对比校验。
- 该步骤仅用于质量验证，不会改变补算结果。
- 当启用缓存且签名命中时，通常直接读取缓存，不会全量重扫。
- 若使用 `--skip-raw-sampling-dist`，则跳过这项校验以减少耗时。

## 4. 输入输出说明

## 4.1 主要输入
- `raw_bayonet_csv_path`：原始过车数据 CSV
- `bayonet_shp_path`：中心城区卡口点位（需含 `KKBH`）
- `route_matched_bayonet_csv_path`：路网可映射卡口清单（需含 `KKBH`）
- `vehicle_table_2021_path`：车辆信息表（优先匹配设备属性）

## 4.2 主要输出
- `preprocessed_data_path`：预处理结果（实际写 Parquet）
- `vehicle_info_path`：规则内车辆信息（实际写 Parquet）
- `excluded_supplement_data_path`：补算结果（实际写 Parquet）
- `excluded_device_attr_path`：设备属性分配明细（实际写 Parquet）
- `hourly_stat_path`：小时流量 Excel
- `travel_time_stat_path`：相邻卡口间隔 Excel

说明：
- 上述 4 个主数据输出会自动规范为 `.parquet` 后缀。
- 统计阶段读取时优先 Parquet，找不到时回退 CSV。

## 5. JSON 配置项中文说明（逐项）

下表对应 `bayonet_data_pipeline.config.example.json` 全字段。

| 字段 | 含义 | 建议/备注 |
|---|---|---|
| `bayonet_shp_path` | 中心城区卡口 shp 路径 | 必须含 `KKBH` |
| `raw_bayonet_csv_path` | 原始过车 CSV 路径 | 超大文件，建议放高速盘 |
| `route_matched_bayonet_csv_path` | 路网可映射卡口表路径 | 必须含 `KKBH` |
| `preprocessed_data_path` | 预处理输出路径 | 自动转 `.parquet` |
| `vehicle_info_path` | 规则内车辆信息输出路径 | 自动转 `.parquet` |
| `hourly_stat_path` | 小时流量统计输出路径 | Excel |
| `travel_time_stat_path` | 相邻卡口间隔统计输出路径 | Excel |
| `excluded_supplement_data_path` | 被排除样本补算输出路径 | 自动转 `.parquet` |
| `excluded_device_attr_path` | 设备属性分配明细输出路径 | 自动转 `.parquet` |
| `vehicle_table_2021_path` | 车辆信息表输入路径 | 优先用于设备级匹配 |
| `raw_csv_chunksize` | 原始 CSV 分块大小 | 大机器可调大（如 100万~500万） |
| `raw_csv_encoding` | 原始 CSV 编码 | `null` 时用 pandas 默认；若乱码可显式 `gbk`/`gb18030` |
| `output_csv_encoding` | CSV 输出编码 | 当前主输出多为 Parquet，保留兼容用途 |
| `parallel_workers` | 阶段1中心城区过滤并行 worker 数 | `0` 表示自动 `CPU-2` |
| `parallel_submit_window` | 阶段1并行在飞任务窗口 | 大值吞吐更高但占内存更多 |
| `monitor_enabled` | 是否输出监控日志 | 建议开启 |
| `monitor_show_resources` | 监控日志是否显示资源占用 | 建议开启 |
| `monitor_every_n_chunks` | 每多少块输出一次监控日志 | 生产可调大减少日志量 |
| `random_seed` | 随机种子 | 保障可复现 |
| `sampling_mode` | 表外设备采样方式 | `joint` 或 `independent` |
| `sampling_dist_cache_enabled` | 是否启用采样分布缓存 | 建议 `true` |
| `sampling_dist_force_rebuild` | 是否强制重建采样分布 | 需要刷新缓存时设 `true` |
| `sampling_dist_parallel_workers` | 采样分布并行 worker 数 | `0` 自动 `CPU-2`，64核机器建议接近 60 |
| `sampling_dist_partition_count` | 采样分布分桶数 | 建议为 worker 的 2~6 倍，默认 256 |
| `sampling_dist_keep_temp` | 是否保留中间分桶文件 | 排障时可设 `true` |
| `sampling_dist_fallback_serial_on_error` | 并行失败时是否回退串行 | 建议 `true` |
| `valid_vehicle_types` | 有效车型编码集合 | 三重规则之一 |
| `car_vehicle_types` | 客车车型集合 | 统计阶段使用 |
| `truck_vehicle_types` | 货车车型集合 | 统计阶段使用 |
| `valid_fuel_types` | 有效燃料编码集合 | 三重规则之一 |
| `local_plate_prefixes` | 本地车牌前缀集合 | 三重规则之一 |
| `travel_time_bins` | 行程时间分箱边界（秒） | 与 `travel_time_labels` 对应 |
| `travel_time_labels` | 行程时间分箱标签 | 数量需为 `len(bins)-1` |

## 6. 推荐配置（64核 + 480G）

建议从以下起步：
- `parallel_workers: 60`
- `parallel_submit_window: 12~24`
- `sampling_dist_parallel_workers: 60`
- `sampling_dist_partition_count: 256`
- `raw_csv_chunksize: 1000000~3000000`
- `sampling_dist_cache_enabled: true`

首次重建采样分布可用：
- `sampling_dist_force_rebuild: true`

重建完成后改回：
- `sampling_dist_force_rebuild: false`

## 7. 常用命令

全流程：
```bash
python bayonet_data_pipeline.py --config ./bayonet_data_pipeline.config.example.json
```

仅补算：
```bash
python bayonet_data_pipeline.py --config ./bayonet_data_pipeline.config.example.json --supplement-excluded-only
```

导出配置模板：
```bash
python bayonet_data_pipeline.py --dump-config-template ./bayonet_data_pipeline.config.example.json
```

诊断统计：
```bash
python bayonet_supplement_diagnostics.py --config ./bayonet_data_pipeline.config.example.json --topk 20 --output-dir ./diag_output
```

## 8. 编码与乱码排查
- 源码与文档建议 UTF-8。
- 外部业务 CSV 需要兼容 `gbk`、`gb18030`、`utf-8-sig`、`utf-8`。
- 读取编码不确定时应打印命中编码，避免静默乱码。
- 终端看到乱码不代表文件损坏，先用 UTF-8 强制读取方式复核文件内容。
