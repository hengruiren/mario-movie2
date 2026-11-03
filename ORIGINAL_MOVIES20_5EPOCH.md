# mario-movie2：原版对照入口，Movies，Stage 2 固定 20% / 最多 8 轮

早停规则：每轮完整验证。验证 loss 严格低于历史最佳值时保存 checkpoint 并清零等待计数；
连续 2 轮没有刷新最低值（相等也算未改善）就停止，最晚第 8 轮结束。最后加载最佳 checkpoint 测试。


源码来自官方 `sunyuanfu/Mario` 提交
`b07993c83a868969b72a9f1eaea823beea0d9500`。官方 README、LICENSE 和全部
22 个 Stage 1 文件保持原样。本说明针对 `scripts/train_movies20_original.sh` 原版对照入口。
默认增强入口另见 `CLASS_FINE_CLIP_CLASS.md`，包含 Class/Fine 和 CLIP Class；没有 Selector 或 V3。

| 项目 | 配置 |
|---|---|
| 数据集 | MAGB Movies，16,672 个节点 |
| 原始划分 | seed=42，60% / 20% / 20% |
| Stage 1 训练节点 | 完整 10,003 个 |
| Stage 1 | 官方双向 InfoNCE，80 轮，batch=500，lr=0.001，wd=0.005 |
| Stage 1 调度 / seed | StepLR 每 10 轮乘 0.95；官方默认训练 seed=123 |
| Stage 1 导出 | 所有节点 `[16672, 2, 512]`，直接交给 Stage 2 |
| Stage 2 训练目标 | 固定 2,001 个，约为 10,003 个训练节点的 20% |
| Stage 2 | 最多 8 轮，连续 2 轮验证 loss 未改善就早停；batch=4，lr=1e-5，wd=0.05，seed=42 |
| Stage 2 邻居池 | 完整 10,003 个训练节点及其训练标签 |
| 邻居排序 / 推理 | 官方每跳 Top-10 / Router 选一路生成 |
| 验证 / 测试 | 完整 3,334 / 3,335 个节点 |
| 模型选择 | 实际训练轮次中验证 loss 最低的 checkpoint，之后在完整测试集生成 |

20% 限制的是 **Stage 2 训练中心节点**，不是全图的 20%，也不缩小邻居标签池。
名单按类别分层抽样，subset seed=42，保存在
`splits/Movies/stage2_train20_seed42.json`；每次验证数据指纹并复用同一份名单。
2,001 来自 `round(10003 * 0.2)`，不是 Grocery 的 2,049。

Stage 1 参数与官方 Movies 一键脚本一致，保留其 seed=123；Stage 2 batch=4
也沿用官方一键脚本。CLI 原始 batch 默认值仍为 3，一键入口显式传入 4。
原始 LLM、LoRA、Projector、Router、三路训练损失、提示词及邻居排名算法均保留。

除训练目标/轮数及记录功能外，有两项运行兼容修复：适配 Transformers 的
`logits_to_keep` / `num_logits_to_keep` 参数名；生成时由 Transformers 管理缓存
解码的 position IDs。`scripts/audit_original.py` 将这两项反向还原后，验证整个
模型文件与官方源码一致。依赖沿用你之前指定的版本，详见 `requirements.txt`。

## 1. 环境

将本目录放到服务器 `/workspace/mario-movie2`。GitHub 仓库：`https://github.com/hengruiren/mario-movie2`。
在 Linux / NVIDIA 服务器上执行：

```bash
cd /workspace/mario-movie2
python3.11 -m venv /workspace/venvs/mario-movie-py311
source /workspace/venvs/mario-movie-py311/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

依赖固定为 torch 2.4.1、DGL 1.1.3+cu121、numpy 1.26.4、Transformers 4.47.1、
PEFT 0.12.0。`requirements-original-cu121.txt` 仅转发同一份依赖。

## 2. 下载 Movies 数据与 Stage 2 模型

已有文件会复用。不需要下载原图、CLIP 或 Llama-3.2 模型；Stage 1 直接使用 MAGB
预提取的 4096D 整图/文本特征。

```bash
export DATA_ROOT=/workspace/data/MAGB
export LLM_MODEL_PATH=/workspace/models/Llama-3.1-8B-Instruct

hf download Sherirto/MAGB \
  --repo-type dataset \
  --revision 73a2a07864bbbffe969fbe349e92e588c3e138da \
  --include \
    'Movies/Movies.csv' \
    'Movies/MoviesGraph.pt' \
    'Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy' \
    'Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy' \
  --local-dir "$DATA_ROOT" \
  --max-workers 4

hf auth login
hf download meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --exclude 'original/*' \
  --local-dir "$LLM_MODEL_PATH" \
  --max-workers 2
```

Movies 使用真实的 `_512_mean.npy` 文本文件，无需 Grocery 的 256 文件名适配。
原始图像特征 dtype=float64、文本特征 dtype=float32，均为 `[16672,4096]`；
官方 Stage 1 会统一转为 float32。

## 3. 运行完整 Stage 1 → Stage 2

先打开 tmux：

```bash
tmux new -s mario-movie
```

在 tmux 内重新激活环境，并设置变量：

```bash
source /workspace/venvs/mario-movie-py311/bin/activate
cd /workspace/mario-movie2

export DATA_ROOT=/workspace/data/MAGB
export LLM_MODEL_PATH=/workspace/models/Llama-3.1-8B-Instruct
export RUN_DIR="$PWD/runs/movies20_original_8epochs_seed42_run1"
export GPU=0

bash scripts/train_movies20_original.sh
```

脚本验证官方源码、固定名单及 CUDA DGL，然后运行完整 80 轮 Stage 1、固定 20%
最多 8 轮 Stage 2（连续 2 轮验证 loss 未改善就早停）。它仅建立运行目录中的数据链接，不修改下载数据。已有 `RUN_DIR`
会拒绝覆盖；再次运行请换一个新目录。

本目录保留了官方的两个 DDP 脚本以供源码对照；它们仍是官方 100% 数据入口；本次也同步设置最多 8 轮、patience=2。
**本次实验使用 `scripts/train_movies20_original.sh`，不要使用旧 DDP 入口。**

## 4. 已有 Movies 原版 InfoNCE embedding，只运行 Stage 2

```bash
source /workspace/venvs/mario-movie-py311/bin/activate
cd /workspace/mario-movie2

DATA_ROOT=/workspace/data/MAGB \
LLM_MODEL_PATH=/workspace/models/Llama-3.1-8B-Instruct \
STAGE1_FEATURE=/absolute/path/to/original/Movies_stage1_mlp.pth \
RUN_DIR="$PWD/runs/movies20_stage2_only_8epochs_seed42_run1" \
GPU=0 \
bash scripts/train_movies20_original.sh
```

外部 embedding 必须来自同一 Movies 节点顺序和原版 InfoNCE Stage 1。
有 provenance sidecar 时会核对方法和哈希；无 sidecar 的旧张量本身无法证明训练来源。
不能拿 Grocery embedding 或之前 Class/Fine、Patch 加权的 embedding 作为本版基线。

直接使用 `stage2/train_llm.py` 时，必须传入
`--movies-subset splits/Movies/stage2_train20_seed42.json --epochs 8 --patience 2`。
不传名单则保留官方全训练划分行为；推荐一律使用上面的一键入口。

## 5. 看进度与结果

```bash
tail -f "$RUN_DIR/stage1/train.log"
tail -f "$RUN_DIR/stage2/train.log"
cat "$RUN_DIR/stage2/history.jsonl"
cat "$RUN_DIR/stage2/metrics.json"
```

官方 Stage 1 不会逐轮打印 loss，末尾会打印 embedding 导出；Stage 2 有进度条。
Stage 2 输出 `best_stage2_llm.pt`、`test_predictions.jsonl`、`metrics.json`，
`run_config.json` 记录实际参数、特征哈希、固定名单哈希与训练中心/邻居池计数。

## 6. 检查源码与运行范围

```bash
python scripts/audit_original.py
python -m unittest discover -s tests -v
bash -n scripts/train_movies20_original.sh
```

本地验证范围见 `VALIDATION.json`。完整 Movies GPU 训练尚未启动，不能将其他数据集
或增强方法的 Accuracy 当作这份原版实验的结果。
