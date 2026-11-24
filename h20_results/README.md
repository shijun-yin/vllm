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

两轮结果表明 Blazedit 和 SPEED RAG 有稳定收益，SPEED coding 仅处于噪声区间。
这是固定请求重放下的本地热态结果，不代表所有线上负载；未将 cold 或单次最优值
作为稳定吞吐结论。两轮 JSON、日志和汇总均以 `refactor_` 为前缀保存在本目录。

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

默认保持 `dual`，4-only/回查只作显式消融。两轮波动较大，不将单轮最高值当作稳定收益。
这些是最新修复后的本地 B4 热态结果，不替代高 batch 复测。对应 JSON、日志和汇总以
`optimized_` 为前缀保存在本目录。

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
