# ModelOpt Checkpoint 在 ms-swift 中继续训练（新增能力说明）

本文档说明本次在 `ms-swift` 中新增的能力：**加载并恢复 ModelOpt checkpoint（含 `modelopt_state.pth`）后继续训练**。

---

## 1. 为什么要加这些代码

### 问题背景

ModelOpt 的 PTQ/QAT 产物通常不仅有权重，还包含 `modelopt_state.pth`。  
这个状态文件记录了量化相关的结构/状态（例如量化器状态）。

### 原本代码为什么不能直接用

原始 `ms-swift` 流程中，模型加载走常规 `from_pretrained` 路径，没有在加载前启用：

- `mto.enable_huggingface_checkpointing()`

结果是：

1. 即使目录里有 `modelopt_state.pth`，也不会自动恢复。
2. 只加载权重时，ModelOpt 结构状态可能缺失，导致“能加载但不是真正恢复了量化状态”。
3. 对量化后继续训练（尤其是需要保持量化状态一致的场景）不可靠。

---

## 2. 本次新增了什么

### 2.1 新增参数

- 文件：`swift/arguments/base_args/model_args.py`
- 新增：`enable_modelopt_hf: bool = False`

作用：显式控制是否启用 ModelOpt HF checkpointing。

### 2.2 参数可从 ckpt 恢复

- 文件：`swift/arguments/base_args/base_args.py`
- 将 `enable_modelopt_hf` 纳入参数恢复列表。

作用：断点恢复/复现实验时不丢这个开关。

### 2.3 模型加载前启用 ModelOpt 恢复

- 文件：`swift/model/register.py`
- 当 `enable_modelopt_hf=True` 时，在加载模型前执行：
  - `mto.enable_huggingface_checkpointing()`

作用：让 `from_pretrained` 自动识别并恢复 `modelopt_state.pth`。

### 2.4 新增一键回归脚本

- 文件：`scripts/smoke_modelopt_hf.sh`

脚本流程：

1. 加载基础模型并做一次 ModelOpt 量化
2. `save_pretrained` 生成 `modelopt_state.pth`
3. 用 `ms-swift` + `enable_modelopt_hf` 重新加载
4. 跑 1 个训练 step 验证恢复后可训练

---

## 3. 如何使用（命令行）

## 3.1 一键 smoke 回归（推荐先跑）

```bash
cd ms-swift
scripts/smoke_modelopt_hf.sh --conda-env modelopt
```

常用可选参数：

- `--model-id Qwen/Qwen2.5-0.5B-Instruct`
- `--model-type qwen2`
- `--quant-cfg INT8_DEFAULT_CFG`
- `--keep-artifacts`

---

## 3.2 在 ms-swift 中继续训练 ModelOpt checkpoint

核心是加上：

- `--enable_modelopt_hf true`

示例（按你本地训练命令改）：

```bash
swift sft \
  --model /path/to/modelopt_checkpoint \
  --model_type qwen2 \
  --enable_modelopt_hf true \
  --tuner_type full \
  --dataset your_dataset \
  --output_dir outputs/modelopt_resume
```

说明：

1. 本地 checkpoint 如自动识别 `model_type` 失败，需手动指定 `--model_type`。
2. `--enable_modelopt_hf` 只负责“恢复 ModelOpt 状态并继续训练”，不等同于在 ms-swift 内新做一套 NVFP4 QAT 流程。

---

## 4. PTQ/QAT 与 ms-swift 的衔接方式（建议）

推荐流程：

1. 在 ModelOpt 仓库做 PTQ 或 QAT，得到 checkpoint（含 `modelopt_state.pth`）
2. 在 ms-swift 用 `--enable_modelopt_hf true` 继续训练

---

## 5. 适用边界

当前新增能力是“**加载/恢复 ModelOpt checkpoint**”层面的支持。  
不是把 ms-swift 原生量化训练参数体系扩展成 `quant_method=nvfp4` 的完整实现。
