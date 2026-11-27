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

## Coding hot 波次的 JIT 根因与固定宽度修复

后续用 vLLM 原生逐请求指标还原了快轨迹：Coding hot-1 共 1054 次 verify、单请求最多
101 轮；原生 iteration 日志为 297 步，GPU elapsed 约 6.198 s、墙钟约 6.248 s，未显示
接受量或尾批异常。`VLLM_GC_DEBUG=1` 在 hot 中只发现毫秒级 GC，数百毫秒回收发生在
进程退出阶段，因此 GC 不是该 hot 停顿根因。

更关键的是默认 JIT monitor 使用 `warning_once`，会隐藏同一函数的新 constexpr/形状变体。
`jit_monitor_verbose=True` 后在 Coding hot-1 明确看到 `_suffix_finalize_kernel` 的
`NB_PO2=1` 编译；完整数据顺序还出现 `NB_PO2=2/4/8/16`。此前“hot 没有 JIT”的判断不成立。

修复将扫描网格继续限制为本轮 `num_scan_blocks`，但 finalize 的归约宽度固定为
`next_power_of_2(self.n_blocks)`，并用本轮块数掩码，同时禁止 `num_blocks` 特化。固定宽度
保持 R5 陈旧 scratch 防护，避免历史长度变化产生新 JIT。42/42 kernel 回归通过，三类数据
所有波次 token 哈希和投机计数与修复前一致。修复后日志只剩一个固定 `NB_PO2=16` 初始化
事件；Coding hot-1/2 分别为 6.159→6.172 s、5.948→5.962 s，热态无额外回退。Blazedit
首次 cold 增加约 1.4 s 是固定 kernel 的首次初始化，不影响 hot。常规第二轮 Coding
hot-1/2 为 6.167/5.946 s，与 verbose 轮一致；相对同一 pure 基线，Blazedit、Coding、
RAG 分别为 **+14.82%、+3.75%、+19.41%**。详见 `coding_fixednb_summary.md`。

## 最新遗留边界与 R4 覆盖闭合

2026-11-26 的代码在上一节基础上补齐独立 suffix 采样后扫描边界、动态小宽度反馈资格，
以及字符串 stop/外部取消请求的跨请求 corpus 排除。正常 EOS、stop token 和 max tokens
仍可入库；相应产品文档和 scheduler 回归已同步。自适应验证的草稿预算再分配不会改变
每请求已验证 token 数，因此独立 suffix 的采样余量保持为 1，不需要额外倾斜余量。

R4 贪心检查改为覆盖完整 token 序列和逐位置 logprob。非零温度真实换尾探针分别覆盖
suffix 接受位置和替换尾部首次拒绝位置；logprob 对照使用同一已提交前缀下实际生成的
纠正 token，而不是已被拒绝、不会作为普通输出返回的 suffix token。sync/async 分别
`1 passed in 115.44s`、`1 passed in 119.77s`，故意扰动回归通过。原始输出保存为
`reviewfix_r4_sync.log` 和 `reviewfix_r4_async.log`。

本轮 suffix/hybrid kernel 为 47/47，通过 scheduler 套件为 306 passed、3 deselected，
standard DFlash 完整 token/logprob E2E 为 1 passed in 101.12s；字符串 stop E2E 已确认
即使设备历史含有客户端不可见尾部，该请求也不会进入共享 corpus，普通请求仍可入库。
10 个变更 Python 文件的 Ruff、格式
及差异空白检查通过。分布式 PP/PCP 和 H20 最终提交复测不在本地单卡覆盖范围内。

### Pooling 完成与 R4 绝对坐标复核

后续提交 `4feaf2e05a559b346935151f688270e0e491becb` 修复普通 pooling 请求完成时
读取空 `sampling_params.stop` 的共享 scheduler 回归，并把 R4 的草稿内偏移改为
`start + draft_offset` 的输出绝对位置。接受位置额外核对该 token 等于实际接受的草稿；
起点前的变化不再被误归为换尾结果。

Pooling 定向回归为 `1 passed in 4.00s`，非零起点、起点前及接受 token 扰动等 CPU
回归为 `5 passed, 8 deselected in 4.80s`。单卡 scheduler 全套为
`307 passed, 3 deselected in 85.43s`，三项排除均因 PP=2/3 需要多张 GPU，未排除时其余
307 项同样通过。

首次修正坐标后的非 batch-invariant sync 在 rejected→target 上出现 0.1011306 的 logprob
中位绝对偏差，未通过原 0.1 阈值。没有提高阈值；启用项目已有的 batch-invariant 执行后，
同前缀、同实际生成 token 的误差归零。最终 sync/async 合并为 `2 passed in 226.35s`：
sync 记录 486 个真实事件，async 记录 851 个，accepted/rejected 对 native 和 target 的
logprob 中位误差均为 0.0，实际比较位置包含 2、3、5、6。原始文件为
`reviewfix2_scheduler.log`、`reviewfix2_r4_sync.log`（数值漂移诊断）、
`reviewfix2_r4_sync_batch_invariant.log`（单项诊断）和
`reviewfix2_r4_sync_async.log`（最终两组）。

下节首先保留 `615057195c` 的历史配对；`4feaf2e05a` 的最新 4070 复跑见随后一节。
最新提交的 H20 三组性能验收及 PP/PCP 实机覆盖仍待执行。

### 615057195c 的 RTX 4070 同期配对

代码提交为 `615057195c71e0dee4080d273c62bc8868015a99`。使用标准 DFlash K=15、j=1、
B4、64K corpus、temperature=0.8、自然 EOS；每类 16 条请求，每波最多 384 token。
每个数据集先独立 warmup 并清空状态，再执行 cold、hot-1、hot-2。第一轮为
pure→splice，第二轮为 splice→pure，四个进程均关闭分段打点。下表吞吐为两轮 hot
合计输出 token 除以合计时间，纯/splice 平均接受量分别为 AL P/S。

| 数据集 | 第一轮 pure/splice tok/s | 第一轮 Delta | 第二轮 pure/splice tok/s | 第二轮 Delta | AL P/S |
| --- | ---: | ---: | ---: | ---: | ---: |
| Blazedit | 742.92 / 847.82 | +14.12% | 742.96 / 847.62 | +14.09% | 6.885 / 8.783 |
| SPEED Coding | 955.40 / 993.93 | +4.03% | 955.68 / 993.39 | +3.95% | 4.566 / 4.839 |
| SPEED RAG | 368.68 / 444.10 | +20.46% | 369.16 / 444.34 | +20.36% | 1.922 / 2.568 |

两组 pure 和 splice 的逐请求 token 哈希、投机计数分别完全一致，组间吞吐差均小于
0.2%。两组 splice 与上一版固定 JIT 结果的所有 cold/hot token 哈希也一致，hot 波次
耗时差在约 0.7% 内。Coding hot-1/hot-2 第一轮为 6.185/5.967 s，第二轮为
6.187/5.971 s，未复现旧版约 8 s 的执行空泡，也不再出现 −10% 到 +3% 的配对波动。
当前证据支持本地 B4 稳定收益，但不外推到 H20、DFlash2 或更高实际并发。

原始文件为 `reviewfix_pair1_pure.{json,log}`、`reviewfix_pair1_splice.{json,log}`、
`reviewfix_pair2_splice.{json,log}`、`reviewfix_pair2_pure.{json,log}`。

### 4feaf2e05a 的 RTX 4070 复跑

在 pooling 与 R4 修复后的最终代码 `4feaf2e05a559b346935151f688270e0e491becb` 上，
按完全相同的模型、采样、B4、独立 warmup、无打点和交替进程顺序再次执行四组进程。
两轮 pure 和 splice 的逐请求 token 哈希、投机计数分别完全一致，也与上节
`615057195c` 的对应模式一致，因此以下差异均来自时间项而不是生成轨迹变化。

| 数据集 | 第一轮 pure/splice tok/s | 第一轮 Delta | 第二轮 pure/splice tok/s | 第二轮 Delta | AL P/S |
| --- | ---: | ---: | ---: | ---: | ---: |
| Blazedit | 731.20 / 847.36 | +15.89% | 739.29 / 843.28 | +14.07% | 6.885 / 8.783 |
| SPEED Coding | 972.37 / 987.22 | +1.53% | 948.92 / 1002.91 | +5.69% | 4.566 / 4.839 |
| SPEED RAG | 376.24 / 448.58 | +19.23% | 370.05 / 444.47 | +20.11% | 1.922 / 2.568 |

Coding 的 splice hot-1 两轮为 6.183/6.190 s，hot-2 为 6.051/5.853 s，未复现旧版约
8 s 慢波次。两轮 pure hot-1/2 分别为 6.180/6.229 s 和 6.369/6.347 s；第二轮纯基线
较慢且 splice hot-2 较快，使配对收益由 +1.53% 变为 +5.69%。Blazedit 第一轮 pure
hot-2 也有 8.553 s 的小幅慢波，另三轮为 8.253–8.354 s。Coding 两次方向均为正，但
增益幅度仍受约 2% 的执行时间漂移影响；Blazedit 和 RAG 的方向及量级较稳定。第二轮 pure
启动时有 TritonBundler cubin 缺失告警，随后正常编译加载并完成预热；两组 pure 的
`_resample_kernel` JIT 均发生在预热阶段，正式 hot 无新增 JIT 告警，不能将启动告警直接
视为 hot 漂移的解释。该结果不外推到 H20、DFlash2 或更高实际并发。

原始文件为 `perf4fe_pair1_pure.{json,log}`、`perf4fe_pair1_splice.{json,log}`、
`perf4fe_pair2_splice.{json,log}`、`perf4fe_pair2_pure.{json,log}`。

### OmniInfer 启发的查询优化验收

本轮首先尝试本地扫描与 corpus 查询的双流并行，但 `overlap_ms` 实测两者交集为 0：
数微秒的首次计算在第二次 Python/Triton launch 提交前已经结束。随后改为单 launch
内的 fused 实现：本地块 program 与 corpus program 同时执行，最后仍按原规则归并。
三 token 全失配预过滤不改变匹配结果，真实重叠区间和候选接受量上下界诊断只用于归因。

48 项 suffix/hybrid kernel 回归通过，另有 60 组固定输入 sequential/fused 对照全部一致。
Fused 的 greedy 为 `1 passed in 181.16s`，R4 sync/async 为 `2 passed in 286.24s`；最终默认
顺序配置复核为 `1 passed in 180.43s`、`2 passed in 277.59s`。最终 R4 sync 486 个事件、
async 851 个事件，accepted/rejected 对 native 和 target 的 logprob 中位误差均为 0。
性能采用两轮三组无打点进程，每组都包含 pure、顺序 splice 和 fused，并交换执行顺序。
两轮 pure、顺序和 fused 各自的逐请求 token 哈希及投机计数完全一致；本轮轨迹与此前
`perf4fe_` 不同，因此只比较同期三组，不套用历史百分比。

| 数据集 | Pure 两轮 tok/s | 顺序 splice 两轮 tok/s | Fused 两轮 tok/s | Fused/顺序两轮 |
| --- | ---: | ---: | ---: | ---: |
| Blazedit | 697.80 / 695.56 | 886.03 / 883.16 | 875.43 / 877.32 | −1.20% / −0.66% |
| SPEED Coding | 884.67 / 880.91 | 888.46 / 886.20 | 892.45 / 890.42 | +0.45% / +0.48% |
| SPEED RAG | 356.14 / 356.80 | 412.35 / 411.57 | 411.03 / 411.90 | −0.32% / +0.08% |

顺序 splice 相对同组 pure 的三类收益方向均成立。Fused 仅在 Coding 上有约 0.46% 的
小幅稳定改善，Blazedit 连续回退，RAG 基本持平，因此最终配置默认使用顺序路径，
`suffix_hybrid_parallel_lookup=true` 仅保留为显式消融。基准脚本对应选项为
`--parallel-lookup`。Fused 会预先计算 corpus，即使本地已有最长匹配；这可能是 Blazedit
不利的一个路径因素，但本轮未单独量化其贡献，不把该解释当作已证实根因。

性能原始文件以 `opt_pair1_`、`opt_pair2_` 为前缀。双流零重叠、顺序/fused 诊断和
候选质量统计以 `opt_smoke_` 为前缀保存；带诊断的轨迹及时间不作为性能结论。

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
