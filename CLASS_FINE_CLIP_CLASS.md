# Movies：Stage 1 Class/Fine + 两阶段之间 CLIP Class

早停规则：每轮完整验证。验证 loss 严格低于历史最佳值时保存 checkpoint 并清零等待计数；
连续 2 轮没有刷新最低值（相等也算未改善）就停止，最晚第 8 轮结束。最后加载最佳 checkpoint 测试。


这份增强方法只位于 `mario-movie2`。`mario-movie` 仍为原版对照。
默认 `scripts/train_movies20.sh` 转到增强入口 `scripts/train_movies20_clip_class.sh`。
原版入口另存为 `scripts/train_movies20_original.sh`。

## 方法与参数

1. 用 MAGB 预提取的 Movies 整图/文本 4096D 特征，运行官方 Stage 1 双向 InfoNCE。
   使用完整 10,003 个训练节点，80 轮，batch=500、lr=0.001、wd=0.005，StepLR 每 10 轮乘 0.95。
   Stage 1 seed=123，与 `mario-movie` 的原版设置保持一致。
2. 根据真实 Movies 类别 ID 和名称，使用 `configs/movies_classes.json` 中的 20 条语义描述。
   冻结 Llama-3.2-11B-Vision-Instruct 提取类别文本 4096D 特征，经过已训练的 Mario Text Encoder
   得到 20 个 512D 类别原型。类别名称顺序与 Movies.csv 一致，不使用 Grocery 描述。
3. 冻结 Text Encoder、类别原型和 InfoNCE logit scale，只微调 Image Encoder 20 轮。
   batch=128、lr=0.0001、wd=0.005，最终使用第 20 轮导出；最佳原型 checkpoint 仅额外保存。
4. 损失为 `L_InfoNCE + 1.0 * L_class + λ_fine * L_fine`。
   Class Loss 是图像与 20 个类别原型的余弦 logits / 0.07 的交叉熵。
   `λ_fine` 在第 1–5 轮为 0，第 6–10 轮为 0.02、0.04、0.06、0.08、0.10，第 11–20 轮为 0.10。
5. Fine Loss 只使用训练节点和训练标签：从微调前 Mario 图像 embedding 中固定检索同类 Top-5，排除自身。
   每张训练图切成 2×2，冻结 CLIP ViT-B/16 编码，计算截断到 [0,1] 的 4×4 余弦矩阵；
   双向 max 后平均作为教师分数。Mario 图像对的原始余弦与该分数做 MSE，不截断学生余弦，保留负值区域梯度。

阶段间 CLIP Class 使用微调后的 Stage 1 导出：

1. 冻结 CLIP 编码全部节点的整图和 4 个裁剪，缓存为 `[N,512]` 和 `[N,4,512]`。
2. 仅用完整 10,003 个训练节点的 CLIP 整图与 Mario 图像向量拟合 affine ridge 映射，ridge=1.0。
   映射不使用类别标签、验证/测试配对；CLIP Patch 通过同一映射转换到 Mario 坐标。
3. 原始 Mario 文字、图片分别与冻结类别原型求余弦并预测类别。
   预测不一致则保持原图像 embedding；一致时以共同预测类原型对 4 个映射后 Patch 做 softmax 加权，温度 0.07。
4. 聚合归一化 Patch，归一化候选结果，并保留原图像向量长度。
   仅当预测类相似度提升大于 1e-6，且自身图文相似度不下降时接受；否则保留原表示。
   决策不使用真实标签，不删除节点，文字 embedding 逐字节保持不变。
5. 导出最终 `[16672,2,512]`，直接作为原版 Stage 2 输入。

这是 **CLIP Patch→映射→Mario 类别引导**，不提取 Llama 图像裁剪，不做 CLIP 预筛。
这次也不使用自身 Text 作为 Patch 权重查询，不启用 V3 或 Neighbor Selector。

| Stage 2 项目 | 保持的配置 |
|---|---|
| 训练中心 | 固定 2,001 个，完整训练划分的约 20% |
| 轮数 / batch / lr | 最多 8（patience=2） / 4 / 1e-5 |
| 邻居池 | 完整 10,003 个训练节点与训练标签 |
| 选择邻居 / 推理 | 原版每跳 Top-10 / Router 选一路生成 |
| 验证 / 测试 | 完整 3,334 / 3,335 个节点 |
| 模型、损失、优化器与选择 | 原版；以验证 loss 最低 checkpoint 做完整测试 |
| 输入 | `RUN_DIR/clip_class/Movies_stage1_mlp.pth` |

Stage 2 网络、损失及推理源码保持不变；CLI 后续按用户要求将轮数 5→8、patience 5→2。
新增模块之前的指纹仍保存在 `STAGE2_BEFORE_EXTENSIONS.json`，审计只允许这两项后续设置变化。
Stage 1 的 22 个官方文件也未修改；增强实现位于新增 `stage1/prototype_patch/`。

## 1. 环境和数据

服务器目录示例 `/workspace/mario-movie2`。在 tmux 中激活环境并重新设置变量：

```bash
tmux new -s mario-movie2
```

```bash
cd /workspace/mario-movie2
source /workspace/venvs/mario-movie-py311/bin/activate

export DATA_ROOT=/workspace/data/MAGB
export LLM_MODEL_PATH=/workspace/models/Llama-3.1-8B-Instruct
export LLAMA_VISION_MODEL_PATH=/workspace/models/Llama-3.2-11B-Vision-Instruct
export CLIP_MODEL_PATH=/workspace/models/clip-vit-base-patch16
export MOVIES_IMAGE_DIR="$DATA_ROOT/Movies/MoviesImages"
export RUN_DIR="$PWD/runs/movies20_clip_class_8epochs_run1"
export GPU=0
```

没有环境时先建立 Python 3.11 环境，再执行：

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

依赖文件保持用户指定版本，不额外覆盖。新方法仍需要冻结 Llama-3.2 生成 20 个类别描述特征，
并需要 CLIP 和原图；MAGB 的整图/节点文本特征直接复用，不重新生成。

已有同一 revision 数据时复用相同路径，否则：

```bash
hf download Sherirto/MAGB \
  --repo-type dataset \
  --revision 73a2a07864bbbffe969fbe349e92e588c3e138da \
  --include \
    'Movies/Movies.csv' \
    'Movies/MoviesGraph.pt' \
    'Movies/MoviesImages.tar.gz' \
    'Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy' \
    'Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy' \
  --local-dir "$DATA_ROOT" --max-workers 4

tar -xzf "$DATA_ROOT/Movies/MoviesImages.tar.gz" -C "$DATA_ROOT/Movies"
test -f "$MOVIES_IMAGE_DIR/0.jpg"
```

已核对压缩包内部目录是 `MoviesImages`。

## 2. 三个模型

使用已获得 Meta 模型下载权限的账号登录。已有完整模型时同路径复用：

```bash
hf auth login

hf download meta-llama/Llama-3.2-11B-Vision-Instruct \
  --revision 9eb2daaa8597bf192a8b0e73f848f3a102794df5 \
  --exclude 'original/*' \
  --local-dir "$LLAMA_VISION_MODEL_PATH" --max-workers 2

hf download openai/clip-vit-base-patch16 \
  --revision 57c216476eefef5ab752ec549e440a49ae4ae5f3 \
  --exclude 'tf_model.h5' 'flax_model.msgpack' \
  --local-dir "$CLIP_MODEL_PATH" --max-workers 2

hf download meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --exclude 'original/*' \
  --local-dir "$LLM_MODEL_PATH" --max-workers 2
```

## 3. 一次运行

```bash
bash scripts/train_movies20.sh
```

等价于 `bash scripts/train_movies20_clip_class.sh`。运行顺序：源码/名单/GPU 检查 →
类别与训练 Patch 缓存 → Stage 1 80+20 → 全节点 CLIP Class → 原版 Stage 2 20% 最多 8 轮、patience=2。
已有 `RUN_DIR` 会拒绝覆盖。缓存带有数据、模型、类别描述及图片指纹；设置不同会报错，不能混用 Grocery 缓存。

Stage 1 seed=123，固定数据划分、抽样和 Stage 2 seed=42；这些含义不同。
此设置与新 Movies 原版基线一致，不能直接拿以前 Grocery seed=42 的结果作为 Movies 提升证据。

## 4. 进度和输出

```bash
tail -f "$RUN_DIR/prepare.log"
tail -f "$RUN_DIR/stage1.log"
tail -f "$RUN_DIR/clip_class/refine.log"
tail -f "$RUN_DIR/stage2/train.log"
cat "$RUN_DIR/stage2/metrics.json"
```

原版 InfoNCE 阶段不逐轮打印 loss；20 轮微调会打印进度及每轮 loss。
主要文件：

- `stage1/base/stage1/encoder.pt`：原版 InfoNCE Encoder，可作为兼容基线 checkpoint 复用。
- `stage1/last_encoder.pt`、`stage1/class_prototypes.pt`：微调模型和冻结类别原型。
- `stage1/Movies_stage1_mlp.pth`：Class/Fine 后、Patch 调整前的 embedding。
- `stage1/stage1_metrics.json`：原版 InfoNCE 与 Class/Fine 后的四个验证集 embedding 指标。
- `clip_class/Movies_stage1_mlp.pth`：Stage 2 最终输入。
- `clip_class/refinement_metrics.json`：CLIP Class 前后指标、接受数量和分组统计。
- `clip_class/node_decisions.jsonl`：每个节点的共识、接受/拒绝原因及权重。
- `stage2/history.jsonl`、`stage2/metrics.json`、`stage2/test_predictions.jsonl`：训练和测试结果。
- `embedding_assets.tar.gz` 和 `embedding_assets_sha256.json`：重用所需的 embedding、Encoder、原型、映射及配置备份。

备份包仅保存在本次运行服务器上；关闭或删除服务器前要复制到持久存储。
以后使用最终 `clip_class/Movies_stage1_mlp.pth` 直接运行 Stage 2，无需重训 Stage 1。
不要把该增强 embedding 传给原版对照脚本，它会拒绝非原版方法的 sidecar。

完整 GPU 训练尚未启动，效果尚未在 Movies 上测得。当前验证范围以 `VALIDATION.json` 为准。
