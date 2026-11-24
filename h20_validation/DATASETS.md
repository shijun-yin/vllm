# Suffix splice 测试数据与口径

本文档对应 `h20_validation/benchmark_splice.py`、`prepare_data.py` 和
`summarize_results.py`。数据集不进入 vLLM 产品 patch；下载后按 SHA256 校验。

## 固定版本

| 数据 | 来源与 revision | 本地文件 SHA256 | 当前用途 |
|---|---|---|---|
| Blazedit 5K | `vdaita/edit_5k_char` @ `c00060332eb3f60ae7b2cd448caa45a83f25f1c3` | `8c284ea22229530d3c1cbcca5448c024adce742d967ddc4ec6727f31fe576154` | 代码整文件改写；高复用正向负载 |
| SPEED-Bench qualitative | `nvidia/SPEED-Bench` @ `fa4e669bee2a4393c56aa8ed05a7c667897f9590` | `bb5ddc59123d0d3aebfb5bd70654dd6b56dde00a44b27153f408616073d08afb` | coding、RAG、writing；分领域对照 |
| Spec-Bench | `hemingkx/Spec-Bench` @ `fd2c1cd7d2201ef71db4c5f4e455008f017967bf` | `4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04` | RAG、summarization；suffix PR 同类负载 |

完整机器可读清单位于 `datasets.json`。下载和校验：

```bash
python h20_validation/prepare_data.py
python h20_validation/prepare_data.py --check-only
```

默认数据根目录是 `/root/qwen35-9b-deploy/data`，可用 `--root` 或
`SPLICE_DATA_ROOT` 修改；当前基准脚本的产品复现路径使用该默认目录。
HF 直连不可用时，可给 `prepare_data.py` 传
`--endpoint https://hf-mirror.com`，模型下载同样通过 `HF_ENDPOINT` 控制。

## 样本选择

所有模式使用同一个 tokenizer、同一组请求 ID 和相同的 prompt token 数。
结果检查器会比较这些字段，不一致就失败。

| 负载 | 选择与过滤规则 | 随机种子 |
|---|---|---:|
| `blazedit` | 打乱后取前 N 条；`0.01 <= norm_distance <= 0.25`；使用原始代码和 change request 的整文件改写模板 | 20261122 |
| `speed_coding` | SPEED-Bench `category=coding`；只使用 `turns[0]` | 20261123 |
| `speed_rag` | SPEED-Bench `category=rag`；只使用 `turns[0]` | 20261123 |
| `speed_writing` | SPEED-Bench `category=writing`；只使用 `turns[0]` | 20261123 |
| `spec_rag` | Spec-Bench `category=rag`；只使用 `turns[0]` | 20261124 |
| `spec_summarization` | Spec-Bench `category=summarization`；只使用 `turns[0]` | 20261124 |

共同规则：

- 默认每类 16 条，prompt 上限 2,200 token，模型上下文 3,072 token。
- 使用各 target 的 chat template，`enable_thinking=False`。
- 默认最多生成 384 token，temperature=0.8、top-p=1.0、top-k=-1；这些是本地
  历史结果口径。H20 四小时最小集显式使用 Qwen3.8 官方默认
  temperature=1.0、top-p=0.95、top-k=20，不做两种采样口径扫描。
- 脚本保留 `--ignore-eos` 作为其他实验的固定工作量选项；本地历史结果和 H20
  最小集都使用自然 EOS，不执行该消融。
- SPEED-Bench 中 `FULL BENCHMARK DATA SHOULD BE FETCHED` 占位行会被排除。
  因 summarization 全是占位内容、roleplay 有效样本不足，本地选择 writing，
  没有把空模板的结果计入性能。
- 实际请求 ID 和 prompt token 数保存在每个结果 JSON 中，不要求读者根据文档猜测。

## 冷态、热态与状态边界

每个数据集独立执行：

1. `reset_spec_state` 清空模型/suffix 路由统计和跨请求语料。
2. `cold`：第一次生成；请求内 prompt 历史可用，但跨请求语料尚未积累。
3. `hot-1`、`hot-2`：按相同请求顺序重放；已完成请求可以进入跨请求语料。
4. 保存 vLLM draft、draft-token、accepted-token 计数和 Hybrid 来源统计。

因此热态结果回答的是“存在可复用历史时是否有收益”，不是随机线上到达流量。
它既包含同请求历史复用，也可能包含同批重放的跨请求复用。生产验收必须补充
不同请求、不同到达顺序和真实多轮 agent 轨迹。

## 指标

- `output_tokens_per_second`：整波实际输出 token 数除以 `LLM.generate` wall time，
  包含该波 prompt 处理，不是纯 decode kernel 时间。
- `accepted_per_draft`：`spec_decode_num_accepted_tokens / spec_decode_num_drafts`，
  不含 target 额外产出的 token。
- `draft_token_acceptance`：接受 token 数除以全部提议 draft token 数。
- `suffix_observed`：实际到达验证位置的 suffix 来源观察数；大于零只能证明换尾
  确实执行，不能单独证明吞吐收益。
- 性能结论必须同时看输出吞吐、接受长度、draft 轮数和重复运行范围。

## 本地 j 选择

使用 Qwen3.5-4B AWQ target、官方 DFlash2、K=7、官方默认采样和 64K 语料，
对六类数据运行 j=1/2/4。`select_j.py` 按两轮热态合计输出 token/s 选值，3%
以内视为噪声并列并选较小 j。六类最终都选择 j=1；Blazedit 的 j=1 明显优于
其他候选，coding 和部分 Spec/SPEED 类别只是并列，不声称精确最优。

机器可读结果在 `local_j_selection.json`，原始结果在本地
`/root/qwen35-9b-deploy/results/local_dflash2_k7_*.json`。该结果是 27B 的先验，
不是对目标 27B DFlash2 的保证；H20 最小集直接使用 j=1，不重复扫描。

## 本地结果的适用范围

本地环境为单张 RTX 4070 12GB。DFlash 主负载做了两次独立 pure/splice 运行；
Spec-Bench 和 MTP7 当前是单次配对结果。温度采样下不同草稿来源会改变随机数
消费过程，相同 seed 不保证逐 token 相同，因此不能用输出逐字一致替代分布验证。

本地还使用过 HumanEval、BFCL v3/v4、SWE-bench Pro V2 等探索性负载；它们不属于
这份 H20 首批固定数据清单。此前 BFCL v4 web/memory 约有 10% 收益，SWE 首步
issue 基本持平。完整 agent 多轮工具执行轨迹仍是后续验收项。

## 数据许可

SPEED-Bench 使用 NVIDIA Evaluation Dataset License，详见源数据集。Blazedit 和
Spec-Bench 的使用条件以各自源仓库为准。本验证包只保存来源、revision 和校验值，
不重新授权或内嵌这些语料。
