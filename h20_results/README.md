# H20 实验保留结果与中断说明

## 状态

旧 H20 系统盘已挂载到新实例，上次未能同步的主性能、并发和语料结果现已恢复到
`previous_h20/results/`。这些文件证明旧实现确实完成了相应实验，但不再代表当前代码。
当前实现的新结果单独保存在 `current_h20/`，避免混用版本。

## 已完成的真实 27B 冒烟

配置：

- target：`Qwen/Qwen3.8-27B-FP8` @ `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`
- drafter：`incoai/Qwen3.8-27B-DFlash2` @ `015e795645c74b1a0eeef3b570031fb62e769bc5`
- K=7、j=1、batch=4、64K corpus、自然 EOS。
- 官方默认采样：temperature=1.0、top-p=0.95、top-k=20。
- Blazedit 和 SPEED coding 各 4 条 prompt、128 token、冷态+一轮热态。

| Dataset | pure tok/s | splice tok/s | Delta | Pure AL | Splice AL | Suffix observed |
|---|---:|---:|---:|---:|---:|---:|
| blazedit | 236.48 | 236.16 | -0.14% | 6.151 | 6.054 | 107 |
| speed_coding | 401.76 | 432.49 | +7.65% | 6.041 | 5.818 | 52 |

no-spec 对应结果：Blazedit 热态约 88.57 token/s，SPEED coding 热态约
107.48 token/s。以上仅为小样本功能冒烟，不作为稳定性能范围或最终调参结论。

已确认：

- 真实 27B FP8、DFlash2 和 K=7 能完成生成。
- 五层 aux hidden states、candidate selector 和 CUDA Graph 正常执行。
- splice 确实有换尾，不是仅加载配置。
- 未出现 aux/context KV、slot mapping、图执行或显存错误。

## 当前 H20 复跑：已确认阶段结果

当前实现使用同一 27B target、DFlash2、K=7、j=1 和官方默认采样，环境、源码与模型均从旧盘复用。
本节 H20 数据对应提交 `35359f54`，早于下节最新边界修复；最新代码的 H20 复测仍需新设备。
batch=4 已完成两轮独立 pure/splice，结果如下：

| Dataset | 第一轮 Delta | 第二轮 Delta | Pure AL | Splice AL |
|---|---:|---:|---:|---:|
| Blazedit | +2.38% | +2.02% | 6.088 | 6.231 |
| SPEED coding | +2.74% | +2.67% | 4.887 | 5.042 |
| SPEED RAG | +4.68% | +4.62% | 3.752 | 3.506 |

batch=1 的 Blazedit/coding/RAG Delta 分别为 +0.08%、-1.83%、-1.80%。
batch=16 分别为 +1.80%、-22.16%、-8.82%；batch=32 分别为 +1.76%、-22.06%、-9.22%。
高并发下 coding/RAG 明显退化，不能用 batch=4 的收益外推。计数显示有效 K 始终为 7，
退化不能简单归因为验证了更多 draft token；历史命中后的查询/选链、候选质量和尾批调度
共同作用，现有整波数据不能给出各 kernel 的精确占比。
语料 0 时 Blazedit/RAG Delta 分别为 -0.07%、+8.41%；128K 时分别为 +1.96%、+4.48%。
Blazedit 的收益依赖跨请求历史，RAG 在仅请求内历史时已有明显复用，扩用语料后收益
仍存在但幅度下降。所有阶段的 JSON、日志和汇总均以 `current_` 为前缀保存在 `current_h20/`。
原生 MTP7 复用同一 27B target，Blazedit Delta 为 +2.30%，SPEED coding 为 -5.15%；
高原生接受率下并不保证 splice 有收益。当前 smoke、batch=4 两轮、并发、语料和 MTP7
实验均已完成并同步，不再有待补的主实验结果。

## 本地重构后的 pure/splice 配对

环境为单张 RTX 4070 12GB，`Qwen3-4B-FP8` + `z-lab/Qwen3-4B-DFlash-b16`，
K=15、j=1、batch=4、64K corpus、temperature=0.8、自然 EOS。每类 16 条请求，
每轮包含 cold 和两轮 hot，以下 Delta 按两轮 hot 合计输出 token/s 计算。

| Dataset | 第一轮 Delta | 第二轮 Delta | Pure AL | Splice AL | Suffix observed |
|---|---:|---:|---:|---:|---:|
| Blazedit | +17.41% | +15.79% | 6.885 | 8.796 | 3476 |
| SPEED coding | -1.62% | +1.26% | 4.566 | 4.658 | 547 |
| SPEED RAG | +27.34% | +22.58% | 1.922 | 2.603 | 1282 |

这两轮曾显示 Blazedit 和 SPEED RAG 收益、SPEED coding 处于噪声区间。
后续打点复核发现历史本地运行存在 Windows GPU 外部占用，因此该表只保留为历史归档，
不再据此宣称稳定收益。两轮 JSON、日志和汇总均以 `refactor_` 为前缀保存在本目录。

## 边界修复与查询优化后的本地配对

后续审查发现的陈旧 scratch、跨轮停止长度和动态 K 归因已修复；新增无效行短路、
空 corpus 跳过、无完整尾部块退出、本地最大匹配短路及最长档投票直达。采样验证
覆盖结构化回退和非结构化真实换尾的 sync/async 分布及 target logprobs。

本地仍使用 RTX 4070、标准 DFlash K=15、B4、64K corpus、temperature=0.8。
以下为两轮 hot 合计吞吐相对 pure 的变化：

| Dataset | 双桶第一轮 | 双桶第二轮 | 4-only | 4 未命中回查 |
|---|---:|---:|---:|---:|
| Blazedit | +28.34% | +9.75% | +5.80% | +10.86% |
| SPEED coding | -2.84% | +0.30% | 未测 | 未测 |
| SPEED RAG | +31.31% | +22.11% | +15.20% | +18.07% |

默认保持 `dual`，4-only/回查只作显式消融。后续复核发现这两组运行存在外部 GPU
占用和 pure 基线组间偏移，不能用该表评价查询优化的稳定收益。对应 JSON、日志和
汇总以 `optimized_` 为前缀保留作问题复现，不作为最终性能结论。

## 波动定位、分段耗时与入库修复

2026-11-25 使用可选 CUDA events 对真实标准 DFlash 路径打点。打点只在
`--profile-stages` 下启用，区间有嵌套，不能求和；最终吞吐另用无打点进程测量。
脚本同时保存逐请求 token 哈希、每波起止时间和 WSL pinned memory 设置。

历史 `optimized_` 两组之间的输出 token/draft 轨迹一致，但 pure 和 splice 的墙钟均
发生组间漂移。复核时还发现 Windows 桌面负载在模型未运行时可占用约 36%–47% GPU。
释放负载后，pure 的三类吞吐重新回到修复前区间，因此跨日期的绝对吞吐不能归因给
splice 代码改动。

固定 greedy、固定 384 token、B4、64K corpus、两轮 hot 的同口径对照表明：
finalize 由旧版约 12.9–15.6 µs/次降至 8.5–9.4 µs/次，说明最长档直达有效；
local scan 约 5.7–7.2 µs/次，剪枝只带来小幅变化，Blazedit 长匹配下额外归约略有
回退。它们是每步微秒级成本，不是此前每波数百毫秒差异的来源。

R8 跨轮长度修复的旧实现会克隆 `total_len` 并逐请求做 GPU 标量赋值。两轮 hot 的
入库 host 时间因此达到 Blazedit 1166.4 ms、Coding 676.5 ms、RAG 850.4 ms；旧版未传
最终长度时只有约 14 ms。修复后最终长度随 `(slot, start, final_length)` 文档描述一次
传入 ingest kernel，缺少长度的旧接口由 kernel 回退到设备历史长度，不再 clone 或逐请求
写标量。修复后的对应 host 时间为 12.5、13.6、12.5 ms。混合截断/回退入库回归和完整
42/42 kernel 测试通过，standard DFlash 多请求 target logprobs E2E 也通过
（1 passed in 105.22s）；无 pinned memory 的 host-only 复核降至每波约 5–6 ms。
host 等待消除不保证等量减少整波墙钟，吞吐以无打点配对为准。host-only 包装本身也可能
改变自然 EOS 轨迹，因此其批次形状只作诊断，不用于宣称慢波次已经修复。

原始 JSON、日志和 GPU 状态分别以 `diagnostic_`、`final_` 为前缀保存。
最终无打点的两次 splice 进程中，三类数据的 cold/hot-1/hot-2 逐请求 token 哈希全部
一致，但 Coding hot-1 仍为 7.958→6.237 s；GPU 频率约 2820 MHz，而利用率由 85.1%
恢复为 99.8%，证实是首轮执行空泡而非候选/输出变化。新增 `--warmup-wave` 将首次
JIT/缓存与正式冷语料波次隔离，warmup 后清空 corpus 和统计，不计入性能汇总。

自然 EOS、无打点、原始采样配置的两轮 hot 合计结果：

| 数据集 | 第一轮相对 pure | 第二轮相对 pure（交换进程顺序） | 独立 warmup 配对 |
| --- | ---: | ---: | ---: |
| Blazedit | +11.29% | +15.66% | +14.43% |
| SPEED Coding | −9.88% | +2.64% | −8.36% |
| SPEED RAG | +19.77% | +20.52% | +19.92% |

warmup 前后的 splice token 哈希仍完全一致，因此 warmup 改变了 JIT 计时边界，但没有
消除 Coding hot-1 空泡；该问题继续由 host-only 逐步计时定位，不能将第二轮单独视为
稳定结论。

## 文件

- `environment.txt`：H20 型号、显存、系统、磁盘与代码状态。
- `setup.log`：完整环境准备、固定模型/数据 revision 和 `SETUP_OK`。
- `h20_smoke_no_spec.{json,log}`：no-spec 真实生成。
- `h20_smoke_pure.{json,log}`：纯 DFlash2 真实生成。
- `h20_smoke_splice.{json,log}`：splice 真实生成与来源统计。
- `h20_smoke_summary.md`：pure/splice 对照汇总。

## 历史结果与当前复跑要求

旧盘的 batch=4 两轮、并发 1/16/32 和语料 0/128K 文件均已恢复；64K 主对照
也有完整 JSON。旧实现的 batch=4 两轮收益约为 2%–4%，高并发下出现明显下降，
详见 `previous_h20/results/h20_summary_*.md`。

当前实现已做正确性修复、检索合并、路由发布合并和 wrapper 拆解，必须重新生成
当前版本结果。每个进程的 JSON 和日志完成后立即同步到 `current_h20/`，不能只凭
旧结果或口头中间日志宣称当前实现通过验收。
