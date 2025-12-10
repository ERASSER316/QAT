# QAT 实验结果分析与改进方案

## 实验结果概览
`evaluate/resultnew` 中的离线评测结果显示：

- **Baseline FP/BF16**：PPL ≈ 1.64，作为控制组。模型大小约 2357 MB，吞吐 ~23.4k tok/s（编译后 ~25.5k tok/s）。
- **PTQ W8A8（baseline_int8）**：PPL ≈ 1.65，与 FP 基线几乎持平，说明离线 PTQ 本身没有明显损失。
- **QAT 后模型（fp-qat）**：PPL ≈ 1.89，即使去掉假量化 wrapper 后的实数权重仍明显劣化。
- **QAT + W8A8 导出（qat_int8）**：PPL ≈ 1.90，性能与 FP 版 QAT 模型相似，劣化明显。

总体结论：现有 QAT 训练未能抵消量化噪声，反而使模型退化。

## 主要问题分析
1. **训练步数与正则不足**：QAT 仅训练 100 步且无 weight decay，量化扰动未被充分吸收，可能导致权重漂移。
2. **缺少在线评测**：训练过程中没有监控 PPL，无法及时发现和定位退化阶段，容易带着坏状态训练到结束。
3. **学习率与优化器收敛不稳定**：QAT 默认学习率 5e-5，未结合量化噪声和额外正则，容易过冲；同时缺少周期性验证支撑的调参依据。

## 改进方案
1. **延长并稳定训练**：默认训练步数提升到 400，并在 AdamW 中加入 weight decay=0.1，配合余弦调度和 warmup，帮助模型在量化扰动下收敛。
2. **加入轻量在线评测**：增加周期性 eval（默认每 50 个更新步，对 test split 取最多 256 条样本）监控 PPL，及时发现退化并指导超参调整。
3. **更保守的学习率初值**：默认学习率降低到 2e-5，与基线更一致，减少早期震荡；如仍退化，可在在线评测反馈下进一步调低或增大 warmup 比例。

## 使用建议
1. 执行 QAT 训练（示例）：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python QAT/train/qat_w8a8_fakequant_train.py \
     --model_path <BF16 基线路径> \
     --data_dir datasets/mergedata_prev2 \
     --save_dir model_qat_w8a8 \
     --train_batch_size 1 \
     --gradient_accumulation_steps 16 \
     --max_steps 400 \
     --lr 2e-5 \
     --weight_decay 0.1 \
     --eval_interval 50 \
     --eval_max_samples 256
   ```
2. 观察日志中的周期性 PPL，若持续升高，可尝试：
   - 进一步降低初始 LR（如 1e-5）或提高 warmup_ratio；
   - 延长训练步数（如 800+）；
   - 检查数据预处理是否保持 label 的 -100 mask，避免无效 token 干扰。
3. 训练完成后，使用 `evaluate/eval_offline.py` 统一评测 PPL 和吞吐，与基线及 PTQ 结果直接对比。
