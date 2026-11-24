# Qwen3.8-27B-FP8 + DFlash2 splice：H20 四小时最小验证集

## 目标与边界

只验证与目标模型和硬件直接相关的内容：

- `Qwen/Qwen3.8-27B-FP8 + incoai/Qwen3.8-27B-DFlash2`、K=7 能真实加载和生成。
- DFlash2 aux hidden states、candidate selector、CUDA Graph 和 splice 确实执行。
- 使用官方默认采样，在 Blazedit、SPEED coding、SPEED RAG 上比较 no-spec、
  纯 DFlash2 和 splice。
- 重点测并发 1/4/16/32；语料只做 0/64K/128K 单变量对照。
- j 已完成本地 Qwen3.5-4B AWQ + 官方 DFlash2、K=7、官方默认采样的
  j=1/2/4 扫描；六类数据集均按热态吞吐和 3% 并列规则选择 j=1。
  H20 不再扫 j，异常时才做 j=1 回退对照。

明确不做：完整模型无关状态矩阵、完整配置回归、多采样参数扫描、
`--ignore-eos`、1M 语料、完整笛卡尔积、多 rank 和 NPU。
自然 EOS 是本轮口径；`--ignore-eos` 虽保留在脚本中，但不进入本最小集。

本目录是实验材料，不属于上游 vLLM 产品代码。数据口径见
[DATASETS.md](DATASETS.md)。

## 本次执行状态

- 当前实现的 27B no-spec/pure/splice 冒烟已完成，K=7、CUDA Graph 和实际换尾通过。
- batch=4 两轮、并发 1/16/32、语料 0/128K 以及原生 MTP7 对照均已完成并同步到
  `../h20_results/current_h20/`；旧盘恢复的历史结果位于 `../h20_results/previous_h20/`。
- 模型、venv、源码和数据均复用旧盘，没有重新下载；每个进程结束后均单独回传结果。
  结果解释和剩余限制见 `../h20_results/README.md`。

## 本地准入门，已完成最小集

上 H20 前必须在本地完成并保存结果：

| 项目 | 当前状态 | H20 是否重复 |
|---|---|---|
| suffix/Hybrid kernel、动态 K、零预算、配置校验 | 40/40 kernel、51/51 配置通过；覆盖合并后的路由发布、有限探索和 DBO 拒绝 | 不重复完整套件 |
| MTP 小模型贪心/logprobs | 原生 MTP 同 K 数值基线、cold/hot corpus 与完整输出检查通过 | 不重复 |
| DFlash2、standard DFlash 真实 LLM | DFlash2 K7、standard DFlash K3 均通过；最终三 drafter 合并回归 3/3 | 不重复 |
| 生命周期、max_tokens、preemption、结构化输出 | max_tokens、结构化 JSON、cold/hot slot 生命周期通过；全注意力 DFlash splice 实际 39 次抢占完成；EOS/stop 未单独枚举 | 不重复 |
| 非零温度分布和 target logprobs | standard DFlash sync/async 各 2000 样本卡方分布检查及逐 token target logprobs，2/2 通过 | 不重复完整分布实验 |
| 自定义/派生 speculator 拒绝、静态检查 | 15 个变更 Python 文件 Ruff/格式通过；复跑 patch 反向 apply 校验通过 | 不重复 |
| 本地 DFlash2 K7、j=1/2/4 数据集扫描 | 已完成，六类均选择 j=1；原始结果和选择 JSON 已保存 | 不重复，只验证 j=1 |

H20 只做短冒烟来发现 27B FP8/backend 特有问题，不用昂贵的 27B 重跑上述
模型无关测试。本地未通过的项目不能因为 H20 时间有限而直接标记为完成。

## 固定版本

| 角色 | 模型 | revision |
|---|---|---|
| target | `Qwen/Qwen3.8-27B-FP8` | `c3a462cc3d38cf351f3b26cbd763469eeb74ef81` |
| drafter | `incoai/Qwen3.8-27B-DFlash2` | `015e795645c74b1a0eeef3b570031fb62e769bc5` |

drafter block size 为 8，所以 K=7。产品 patch 基于 vLLM `bf151b7eb1`。

## 文件

- `suffix_splice.patch`：产品代码、测试与产品文档。
- `benchmark_splice.py`：no-spec / pure / splice 同口径离线基准。
- `prepare_data.py`、`datasets.json`：固定数据 revision 和 SHA256。
- `summarize_results.py`：检查对照配置、请求 ID、prompt token 和实际换尾。
- `local_j_sweep.sh`、`select_j.py`、`local_j_selection.json`：本地 K=7 j 扫描、
  选择规则和结果。
- `DATASETS.md`：样本和指标说明。

## 0. 准备，约 30 分钟

假设材料位于 `/root/qwen35-9b-deploy/h20_validation`：

```bash
export ROOT=/root/qwen35-9b-deploy
export VALIDATION=$ROOT/h20_validation
export VLLM_SRC=$ROOT/vllm-h20
export PY=$ROOT/.venv-h20/bin/python
export PATH=$HOME/.local/bin:$PATH

git clone https://github.com/vllm-project/vllm.git "$VLLM_SRC"
git -C "$VLLM_SRC" checkout bf151b7eb1
git -C "$VLLM_SRC" apply --check "$VALIDATION/suffix_splice.patch"
git -C "$VLLM_SRC" apply "$VALIDATION/suffix_splice.patch"

uv venv --python 3.12 "$ROOT/.venv-h20"
cd "$VLLM_SRC"
VLLM_USE_PRECOMPILED=1 uv pip install \
  --python "$PY" -e . --torch-backend=auto
uv pip install --python "$PY" -r requirements/test/cuda.in

export TARGET=Qwen/Qwen3.8-27B-FP8
export DRAFTER=incoai/Qwen3.8-27B-DFlash2
export TARGET_REV=c3a462cc3d38cf351f3b26cbd763469eeb74ef81
export DRAFTER_REV=015e795645c74b1a0eeef3b570031fb62e769bc5
mkdir -p "$ROOT/results"
$PY "$VALIDATION/prepare_data.py"
$PY "$VALIDATION/prepare_data.py" --check-only
```

网络需要时设置 `HF_ENDPOINT=https://hf-mirror.com`；Xet 异常可设置
`HF_HUB_DISABLE_XET=1`。权重未预置且下载缓慢时，准备阶段可能超过 30 分钟，
这部分不能通过压缩正确性测试来补回。

后续命令都使用官方默认采样 `1.0/0.95/20`，自然 EOS：

```bash
COMMON=(
  --family dflash
  --target-model "$TARGET" --target-revision "$TARGET_REV"
  --draft-model "$DRAFTER" --draft-revision "$DRAFTER_REV"
  --num-speculative-tokens 7 --prefix-len 1
  --temperature 1.0 --top-p 0.95 --top-k 20
  --gpu-memory-utilization 0.90
)
DATASETS=blazedit,speed_coding,speed_rag
```

## 1. 真实 27B 冒烟，约 30 分钟

依次运行三个独立进程：

```bash
for mode in no_spec pure splice; do
  $PY "$VALIDATION/benchmark_splice.py" "${COMMON[@]}" \
    --mode "$mode" --datasets blazedit,speed_coding \
    --num-prompts 4 --max-tokens 128 --hot-rounds 1 --batch-size 4 \
    --output "$ROOT/results/h20_smoke_${mode}.json" \
    > "$ROOT/results/h20_smoke_${mode}.log" 2>&1 || exit 1
done

$PY "$VALIDATION/summarize_results.py" \
  "$ROOT/results/h20_smoke_pure.json" \
  "$ROOT/results/h20_smoke_splice.json"
```

通过条件：

- 日志确认 FP8 target、`DFlash2DraftModel`、K=7 和五层 aux hidden states。
- no-spec、pure、splice 都完成真实生成，pure 的 draft/accepted 计数非零。
- splice 每类 `suffix_observed>0`，不是只启动服务而从未换尾。
- 没有 aux/context KV、CUDA Graph、slot mapping、越界或运行时 OOM。
- 保存启动日志、实际显存、KV token 和图捕获信息。

若纯 DFlash2 都不能运行，停止性能实验，先修模型/backend；若 pure 正常而
splice 失败，只排查 Hybrid，不扩大参数矩阵。

## 2. 主对照，约 60 分钟

batch=4、j=1、64K 语料。pure/splice 做两次独立运行；时间允许再补第三次。
no-spec 做一次整波基线。

```bash
$PY "$VALIDATION/benchmark_splice.py" "${COMMON[@]}" \
  --mode no_spec --datasets "$DATASETS" \
  --num-prompts 16 --max-tokens 384 --hot-rounds 2 --batch-size 4 \
  --output "$ROOT/results/h20_no_spec_b4.json" \
  > "$ROOT/results/h20_no_spec_b4.log" 2>&1

for run in 1 2; do
  for mode in pure splice; do
    $PY "$VALIDATION/benchmark_splice.py" "${COMMON[@]}" \
      --mode "$mode" --datasets "$DATASETS" \
      --num-prompts 16 --max-tokens 384 --hot-rounds 2 --batch-size 4 \
      --output "$ROOT/results/h20_${mode}_b4_r${run}.json" \
      > "$ROOT/results/h20_${mode}_b4_r${run}.log" 2>&1 || exit 1
  done
  $PY "$VALIDATION/summarize_results.py" \
    "$ROOT/results/h20_pure_b4_r${run}.json" \
    "$ROOT/results/h20_splice_b4_r${run}.json" \
    | tee "$ROOT/results/h20_summary_b4_r${run}.md"
done
```

两次结果报告范围，不伪造中位数或稳定性。若两次方向相反，优先用剩余时间补
第三次，不做 j 消融。

## 3. 并发扫描，约 45–60 分钟

固定 j=1、64K 语料、官方采样。batch=4 已有主对照，只补 1/16/32，每个配置
一次 pure/splice 配对：

```bash
for batch in 1 16 32; do
  for mode in pure splice; do
    $PY "$VALIDATION/benchmark_splice.py" "${COMMON[@]}" \
      --mode "$mode" --datasets "$DATASETS" \
      --num-prompts 16 --max-tokens 384 --hot-rounds 2 --batch-size "$batch" \
      --output "$ROOT/results/h20_${mode}_b${batch}.json" \
      > "$ROOT/results/h20_${mode}_b${batch}.log" 2>&1 || exit 1
  done
  $PY "$VALIDATION/summarize_results.py" \
    "$ROOT/results/h20_pure_b${batch}.json" \
    "$ROOT/results/h20_splice_b${batch}.json" \
    | tee "$ROOT/results/h20_summary_b${batch}.md"
done
```

注意 batch=32 时每波只有 16 条请求，它用于验证调度上限和固定 K 成本，不代表
持续 32 并发饱和服务。记录日志中的真实 running 请求数。所有单轮并发结果标为
探索性，不宣称稳定范围。

## 4. 语料最小消融，约 20–30 分钟

只在 batch=4、Blazedit 和 SPEED RAG 上比较 0 与 128K；64K 已由主对照提供：

```bash
for corpus in 0 131072; do
  for mode in pure splice; do
    $PY "$VALIDATION/benchmark_splice.py" "${COMMON[@]}" \
      --mode "$mode" --datasets blazedit,speed_rag \
      --corpus-tokens "$corpus" --num-prompts 16 --max-tokens 384 \
      --hot-rounds 2 --batch-size 4 \
      --output "$ROOT/results/h20_${mode}_corpus${corpus}.json" \
      > "$ROOT/results/h20_${mode}_corpus${corpus}.log" 2>&1 || exit 1
  done
done
```

纯模型不使用语料，但保持相同脚本参数和批次路径。主要比较 splice 的 0/64K/128K，
区分请求内复用、跨请求收益和显存开销，不测 1M。

## 5. 本地 j 扫描结果，H20 不重复

本地使用 AWQ Qwen3.5-4B target、官方 DFlash2、K=7、官方默认采样、64K 语料和
batch=4。每个 j 跑冷态和两轮热态；按合计热态输出 token/s 比较，3% 以内视为
并列并选择较小 j：

| 数据集 | j=1 token/s | j=2 token/s | j=4 token/s | 选值与解释 |
|---|---:|---:|---:|---|
| Blazedit | 611.53 | 591.50 | 560.23 | j=1，明显最优 |
| SPEED coding | 577.26 | 572.61 | 580.25 | 噪声并列，选 j=1 |
| SPEED RAG | 378.99 | 373.91 | 376.46 | 噪声并列，选 j=1 |
| SPEED writing | 350.19 | 331.78 | 332.84 | j=1 |
| Spec-Bench RAG | 364.23 | 373.88 | 370.83 | 噪声并列，选 j=1；三者均低于 pure |
| Spec-Bench summarization | 359.38 | 363.58 | 358.29 | 噪声并列，选 j=1 |

完整数值、接受长度、来源观察和 pure 对照在 `local_j_selection.json`。这是单轮
本地扫描，不宣称对所有模型普遍最优；j 也依赖 drafter 和 K。H20 首批三个数据集
直接使用 j=1。只有目标机上出现与本地明显不同的换尾/接受异常，才增加 j=1
回退对照，不再默认扫描 j=2/4。

## 6. 排查预留，约 60 分钟

优先处理阻塞问题：模型/backend 加载、aux hidden states、DFlash2 graph、
Hybrid 越界或状态错误。只有出现“功能正确但持续性能回退”才 profile：

- DFlash2 forward、candidate selector；
- suffix lookup、route/publish；
- target verify、真实提交量和实际图档。

若需要修改模型无关的通用状态逻辑，回到本地小模型回归，不在 H20 上反复跑
完整 27B 状态矩阵。

## 结果判断与归档

最小通过标准：

- 27B 三路真实生成成功，DFlash2/K=7 和实际 splice 均有直接证据。
- batch=4 的两次 Blazedit pure→splice 结果方向一致；若不一致，补第三次或标为
  无结论，不能选择性汇报。
- 同时报告 no-spec、pure、splice，接受长度和整波输出 token/s。
- 并发 1/4/16/32 全部列出，包括负收益；单轮结果只作探索性判断。
- 语料和 j 的单变量结果不得外推成完整参数最优点。

保存代码与 patch SHA256、模型/数据 revision、完整命令、原始 JSON、日志、
汇总和 profiler（如有）。明确区分本地已验证、H20 已验证、单次观察、重复结果、
单 rank 和未覆盖范围。
