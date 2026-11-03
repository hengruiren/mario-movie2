# mario-movie2：Movies 20% / 最多 8 轮，Class/Fine + CLIP Class

早停规则：每轮完整验证。验证 loss 严格低于历史最佳值时保存 checkpoint 并清零等待计数；
连续 2 轮没有刷新最低值（相等也算未改善）就停止，最晚第 8 轮结束。最后加载最佳 checkpoint 测试。


本目录已加入 Stage 1 的 Class Loss、Fine Loss，以及 Stage 1/2 之间的 CLIP Class 加权。
`mario-movie` 保持原版对照，不受本次修改影响。

**默认入口 `scripts/train_movies20.sh` 现在运行增强版本。**
完整参数、下载和运行流程见 [CLASS_FINE_CLIP_CLASS.md](CLASS_FINE_CLIP_CLASS.md)。

- Stage 1：完整 10,003 个训练节点；原版 InfoNCE 80 轮 + Image-only 联合微调 20 轮。
- 微调：λ_class=1.0，λ_fine 先为 0，再逐步升到 0.1；Text Encoder/类别原型冻结。
- 中间处理：CLIP→训练集拟合的 Mario 空间映射→共同预测类别引导四个 Patch 加权→双条件接受。
- Stage 2：仅更换输入 embedding；保留原版 20%（2,001 个中心）、最多 8 轮（patience=2）、batch=4、Top-10、Router。
- Stage 2 网络、损失和推理不变；CLI 仅同步调整最多 8 轮、patience=2，没有 V3 或 Neighbor Selector。
- requirements.txt 保持用户指定版本不变。

已有完整数据和模型时：

```bash
source /workspace/venvs/mario-movie-py311/bin/activate
cd /workspace/mario-movie2
export DATA_ROOT=/workspace/data/MAGB
export LLM_MODEL_PATH=/workspace/models/Llama-3.1-8B-Instruct
export LLAMA_VISION_MODEL_PATH=/workspace/models/Llama-3.2-11B-Vision-Instruct
export CLIP_MODEL_PATH=/workspace/models/clip-vit-base-patch16
export MOVIES_IMAGE_DIR="$DATA_ROOT/Movies/MoviesImages"
export RUN_DIR="$PWD/runs/movies20_clip_class_8epochs_run1"
export GPU=0
bash scripts/train_movies20.sh
```

如需运行原版对照，显式使用 `scripts/train_movies20_original.sh`，原版说明保存在
[ORIGINAL_MOVIES20_5EPOCH.md](ORIGINAL_MOVIES20_5EPOCH.md)。
完整 GPU 实验尚未启动，验证范围以 `VALIDATION.json` 为准。
