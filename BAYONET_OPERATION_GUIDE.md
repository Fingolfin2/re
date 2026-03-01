# bayonet_data_pipeline 运行手册（命令、效果、配置）

## 1. 适用范围
- 主流程脚本：`bayonet_data_pipeline.py`
- 诊断脚本：`bayonet_supplement_diagnostics.py`
- 配置文件：`bayonet_data_pipeline.config.example.json`

本手册用于快速执行任务、理解日志效果、调整配置并定位常见问题。

## 2. 运行前检查
1. 确认配置文件路径可读，且输入文件存在：
   - `raw_bayonet_csv_path`
   - `bayonet_shp_path`
   - `route_matched_bayonet_csv_path`
   - `vehicle_table_2021_path`
2. 确认环境具备 Parquet 引擎：`pyarrow` 或 `fastparquet`。
3. 建议在 `re/` 目录执行命令。

## 3. 常用命令与效果

### 3.1 仅执行补算阶段（常用）
```bash
python bayonet_data_pipeline.py --config ./bayonet_data_pipeline.config.example.json --supplement-excluded-only
```
效果：
- 读取中心城区记录并切分 `included/excluded`。
- 仅对 `excluded` 做映射+属性重建。
- 输出：
  - `center_excluded_supplement_df.parquet`
  - `中心城区补算设备属性分配明细.parquet`
- 日志会显示：
  - 补算设备总数
  - 采样分配设备数
  - 补算记录数
  - 阶段耗时与资源使用

### 3.2 全流程执行（预处理 + 统计 + 补算）
```bash
python bayonet_data_pipeline.py --config ./bayonet_data_pipeline.config.example.json
```
效果：
- 先输出预处理数据与车辆信息表（Parquet）
- 再输出统计 Excel
- 最后执行补算

### 3.3 导出配置模板
```bash
python bayonet_data_pipeline.py --dump-config-template ./bayonet_data_pipeline.config.example.json
```
效果：
- 生成一份完整配置模板，便于迁移与参数管理。

### 3.4 运行补算诊断统计（完整）
```bash
python bayonet_supplement_diagnostics.py --config ./bayonet_data_pipeline.config.example.json --topk 20 --output-dir ./diag_output
```
效果：
- 输出清洗损失构成（dropna vs dedup）
- 输出去重键冲突统计
- 输出未映射 KKBH Top 分布
- 对比补算后 CLLX/RLLX 分布（vs included、vs 车辆表、vs 原始CSV去重分布）
- 统计最终结果本地车牌占比（记录级、设备级）
- 明细 CSV 写入 `diag_output/`

### 3.5 运行补算诊断统计（快速）
```bash
python bayonet_supplement_diagnostics.py --config ./bayonet_data_pipeline.config.example.json --skip-raw-sampling-dist --output-dir ./diag_output
```
效果：
- 跳过“补算结果 vs 原始CSV去重分布”的对比基准构建，显著减少耗时。
- 仍会输出：清洗损失、去重冲突、未映射 KKBH、补算分布 vs included、补算分布 vs 车辆表、本地车牌占比等核心结果。

### 3.6 快速诊断 vs 完整诊断
- 完整诊断（不加 `--skip-raw-sampling-dist`）：
  - 会生成 `dist_compare_supplement_vs_raw.csv`。
  - 这是质量校验基准，不会改变补算结果本身。
- 快速诊断（加 `--skip-raw-sampling-dist`）：
  - 不生成 `dist_compare_supplement_vs_raw.csv`。
  - 速度更快，适合日常巡检。

## 4. 关键配置注释（重点字段）

### 4.1 基础执行参数
- `raw_csv_chunksize`：CSV 分块大小。大机器可调大。
- `parallel_workers`：阶段1并行 worker，`0` 为自动 `CPU-2`。
- `parallel_submit_window`：阶段1在飞任务窗口，越大吞吐可能更高，但更占内存。
- `monitor_every_n_chunks`：监控日志间隔。

### 4.2 采样相关参数
- `sampling_mode`：
  - `joint`：按 `(CLLX,RLLX)` 联合分布采样。
  - `independent`：按边际分布独立采样。
- `random_seed`：采样随机种子，确保可复现。

### 4.2.1 当前默认口径（已确认）
- 当前实现默认是“真实还原原始总体分布”：
  - 采样分布直接来自原始 CSV 清洗+去重后的联合分布。
  - 允许出现 `C/O/AC`、新能源、非常规车型等编码。
- 该口径下无需改代码。
- 如果要改成“只保留研究规则口径（如仅 K/H + A/B）”，属于业务逻辑变更，需先确认再改。

### 4.3 采样分布并行/缓存参数（新增）
- `sampling_dist_cache_enabled`：是否启用缓存。
  - `true`：命中缓存直接读取，避免重复全量扫描。
- `sampling_dist_force_rebuild`：是否强制重建缓存。
  - `true`：忽略缓存，重新计算。
  - 首次压测或原始数据更新后建议设 `true`，后续改回 `false`。
- `sampling_dist_parallel_workers`：采样分布聚合并行 worker。
  - `0` 为自动 `CPU-2`。
- `sampling_dist_partition_count`：采样分布分桶数。
  - 建议为 worker 的 2~6 倍（64核节点常用 256）。
- `sampling_dist_keep_temp`：是否保留中间分桶文件。
  - 排障时可设 `true`。
- `sampling_dist_fallback_serial_on_error`：并行失败时是否回退串行。
  - 建议保持 `true` 保证流程可用。

### 4.4 规则参数
- `valid_vehicle_types`：有效车型集合。
- `valid_fuel_types`：有效燃料集合。
- `local_plate_prefixes`：本地车牌前缀集合。

## 5. 日志关键字与效果解读

### 5.1 `sampling-dist`
- `phase=partition`：分桶阶段，主要是清洗并按哈希写分桶。
- `phase=aggregate, mode=parallel`：并行聚合分桶阶段。
- `done_buckets=x/y`：分桶聚合进度。
- `dedup_rows`：去重后累计样本量。

### 5.2 `采样分布缓存已更新`
- 说明本次重算已成功落缓存。
- 下次在签名不变情况下可直接命中缓存。

### 5.3 结果日志
- `补算设备总数`：参与补算的设备数。
- `采样分配设备`：不在车辆表内、由采样赋值的设备数。
- `补算记录数（仅 excluded）`：补算结果记录条数。

## 6. 64核+480G 推荐参数
建议起步：
- `parallel_workers: 60`
- `parallel_submit_window: 12~24`
- `sampling_dist_parallel_workers: 60`
- `sampling_dist_partition_count: 256`
- `raw_csv_chunksize: 1000000~3000000`
- `sampling_dist_cache_enabled: true`

首次重建缓存：
- `sampling_dist_force_rebuild: true`

后续常规运行：
- `sampling_dist_force_rebuild: false`

## 7. 常见问题

### 7.1 报错停在 `pd.read_csv(...)`
可能原因：
- 原始 CSV 编码与配置不匹配。
- 文件路径或挂载异常。
- 文件损坏/截断。

处理建议：
1. 检查 `raw_bayonet_csv_path` 是否可读。
2. 显式设置 `raw_csv_encoding` 为 `gbk` 或 `gb18030` 重试。
3. 用小样本先验证读取。

### 7.2 终端显示中文乱码
说明：
- 多数是终端编码显示问题，不一定是文件损坏。
处理建议：
1. 保持源码/文档 UTF-8。
2. 外部 CSV 按 `gbk/gb18030/utf-8-sig/utf-8` 回退读取。
3. 用“强制 UTF-8 读取”方式复核文件内容。

### 7.3 出现 `Stopped` 挂起任务
处理：
- 查看：`jobs -l`
- 继续前台：`fg %1`
- 结束任务：`kill %1`

## 8. 输出文件清单（补算场景）
- 补算结果：
  - `.../center_excluded_supplement_df.parquet`
- 设备属性明细：
  - `.../中心城区补算设备属性分配明细.parquet`
- 采样分布缓存：
  - `.../raw_sampling_distribution_cache.parquet`
  - `.../raw_sampling_distribution_cache.meta.json`
- 诊断脚本输出（若设置 `--output-dir`）：
  - `diagnostics_summary.csv`
  - `dedup_key_conflict_groups_top.csv`
  - `excluded_unmapped_kkbh_top.csv`
  - `dist_compare_supplement_vs_included.csv`
  - `dist_compare_supplement_vs_vehicle.csv`
  - `dist_compare_supplement_vs_raw.csv`（未跳过 raw 时）
